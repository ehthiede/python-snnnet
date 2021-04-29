"""This module contains the main utilities needed for the training process."""

import os
import pickle
import time

import torch

from snnnet import distributed_utils
from snnnet.molecule import molecule_vae
from snnnet.utils import human_units


def _copy_to_device(batch, device):
    return {
        k: v.to(device, non_blocking=True)
        for k, v in batch.items()
    }


def _detach_all(batch):
    return {
        k: v.detach() for k, v in batch.items()
    }


class PerformanceMeasure:
    """Utility class for measuring performance.

    This class tracks the elapsed time and number of processed examples.
    """
    def __init__(self):
        self.reset()

    def record_processed(self, num_examples):
        self._num_examples += num_examples

    def reset(self):
        self._last_time = time.perf_counter()
        self._num_examples = 0

    def get_counter(self, reset=True):
        elapsed = time.perf_counter() - self._last_time
        num_examples = self._num_examples

        if reset:
            self.reset()
        return elapsed, num_examples


class Accumulator:
    """Utility class to accumulate a dictionary of tensors.

    This class can be used to compute the average of a dictionary of tensors.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.values = {}
        self.total_weight = 0

    def record(self, values, weight=1):
        self.total_weight += weight

        for k, v in values.items():
            self.values[k] = v.type(torch.DoubleTensor) * weight + self.values.get(k, 0)

    def get_average(self, reset=True):
        averages = {
            k: v / self.total_weight
            for k, v in self.values.items()
        }

        if reset:
            self.reset()

        return averages

    def reduce(self, target_rank=0, group=torch.distributed.group.WORLD):
        for k in sorted(self.values):
            self.values[k] = self.values[k].cuda()
            torch.distributed.reduce(self.values[k], target_rank, group=group)
        total_weight = torch.tensor(self.total_weight).cuda()
        torch.distributed.reduce(total_weight, target_rank, group=group)
        self.total_weight = total_weight.item()


def symmetrize_graph(E, ensure_boolean=True, dim0=1, dim1=2):
    """Ensures that the specified adjacency matrix is symmetric.
    """
    E_t = torch.transpose(E, dim0, dim1)
    if ensure_boolean:
        E_symm = torch.abs(E) + torch.abs(E_t)
        E_symm = (E_symm > 0).float()
    else:
        E_symm = E + E_t
    return E_symm


def check_number_incorrect_edges(predicted, E_true):
    batch_size = E_true.shape[0]
    E_discretized = (torch.sigmoid(predicted) > 0.5).float()

    # Find the incorrect edges.
    misplaced_bond_matrix = torch.abs(E_discretized - E_true)
    edge_is_incorrect = (misplaced_bond_matrix > 0.).float()
    mean_number_incorrect_edges = torch.sum(edge_is_incorrect) / batch_size

    # Check how many graphs are completely correct.
    graph_is_incorrect = torch.tensor([torch.sum(E_i) > 0 for E_i in edge_is_incorrect])
    num_incorrect_graphs = torch.sum(graph_is_incorrect.float())
    fraction_incorrect_graphs = num_incorrect_graphs / batch_size
    return fraction_incorrect_graphs, mean_number_incorrect_edges


class Harness:
    def __init__(self, train_directory, model, optim, scheduler=None, loss_weights=None,
                 summary=None, device='cuda', amp_enabled=False, profile_enabled=False,
                 train_sampler=None, dist_config=None):
        """Initializes a new training harness for training the VAE.

        Parameters
        ----------
        train_directory : str
            The directory containing all the training logs.
        model : torch.nn.Module
            A module representing the model
        optim : torch.optim.Optimizer
            The optimizer to use for the model
        scheduler
            Optional torch scheduler to use for controlling the learning rate
        loss_weights : dict
            Dictionary with loss component weights. Allowed keys are 'pos', 'edge', 'node', and 'kl',
            with values used according to `vae_loss_function`
        summary : torch.utils.tensorboard.SummaryWriter, optional
            An optional summary writer to record training summaries.
        device : torch.device
            The torch device to use for the model.
        amp_enabled : bool
            Indicates whether mixed precision training should be enabled.
        profile_enabled : bool
            Indicates whether profiling information should be saved at each epoch.
        train_sampler : torch.utils.data.distributed.DistributedSampler, optional
            If not None, the distributed sampler used for training. Its epoch will
            be set before obtaining the dataloader.
        dist_config : distributed_utils.DistributedTrainingInfo, optional
            If not None, configuration for distributed (multi-gpu) training.
        """

        self.amp_enabled = amp_enabled
        self.profile_enabled = profile_enabled
        self.model = model.to(device)
        self.optim = optim
        self.scheduler = scheduler
        self.device = device
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.train_directory = train_directory

        if loss_weights is None:
            loss_weights = {}

        self.loss_weights = {
            'pos': 1,
            'edge': 1,
            'node': 1,
            'kl': 1
        }

        for k in self.loss_weights:
            if k in loss_weights:
                self.loss_weights[k] = loss_weights[k]

        self.summary_writer = summary

        self.global_step = 0
        self.perf_measure = PerformanceMeasure()
        self.metrics_accumulator = Accumulator()
        self.train_sampler = train_sampler
        self.dist_config = dist_config

    def log(self, fmt, *args, **kwargs):
        if not distributed_utils.is_leader(self.dist_config):
            return
        print(fmt.format(*args, **kwargs))

    def train_epoch(self, dataloader, epoch_index):
        self.model.train()
        self.perf_measure.reset()

        self.log('Starting training for epoch {}', epoch_index + 1)
        epoch_start = time.perf_counter()
        total_examples = 0

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch_index)

        for i, batch in enumerate(dataloader):
            num_examples = batch['node_features'].shape[0]
            if self.dist_config:
                num_examples *= self.dist_config.world_size

            batch = _copy_to_device(batch, self.device)

            self.train_single(batch, epoch_index, i)

            total_examples += num_examples
            self.global_step += num_examples
            self.perf_measure.record_processed(num_examples)

        if self.scheduler is not None:
            self.scheduler.step()

        elapsed = time.perf_counter() - epoch_start

        self.log('Training done for epoch {} in {:.0f} s ({}mol / s)',
                 epoch_index + 1, elapsed, human_units(total_examples / elapsed))

    def train_single(self, batch, epoch_index, batch_index):
        self.optim.zero_grad()

        with torch.autograd.profiler.profile(enabled=self.profile_enabled and batch_index == 29, use_cuda=True) as prof:
            losses = self.run_single_batch(batch)
            self.grad_scaler.scale(losses['total_loss']).backward()
            self.grad_scaler.step(self.optim)
            self.grad_scaler.update()

        losses = _detach_all(losses)

        if prof is not None:
            profile_save_dir = os.path.join(self.train_directory, 'profile')
            os.makedirs(profile_save_dir, exist_ok=True)
            profile_file = os.path.join(profile_save_dir, 'profile_{}_{}.pkl'.format(epoch_index + 1, batch_index + 1))
            print('Writing profile information to path {}'.format(profile_file))
            with open(profile_file, 'wb') as f:
                pickle.dump(prof, f)

        self.metrics_accumulator.record(losses, weight=batch['node_features'].shape[0])

        if (batch_index + 1) % 20 == 0:
            self.record_summary(_copy_to_device(losses, 'cpu'), epoch_index, batch_index, batch['node_features'].shape[0])

    def run_single_batch(self, batch):
        edge_features = batch['edge_features']
        node_features = batch['node_features']
        positional_features = batch['positional_features']

        with torch.cuda.amp.autocast(enabled=self.amp_enabled):
            (logit_edges, logit_nodes), mu, logvar = self.model(edge_features, node_features, positional_features)
            total_loss, node_loss, edge_loss, kl_loss = molecule_vae.vae_loss_function(
                logit_edges, edge_features, mu, logvar,
                self.loss_weights['pos'], self.loss_weights['edge'],
                logit_nodes, batch['node_features'],
                self.loss_weights['node'], self.loss_weights['kl'])

        fraction_incorrect, mean_num_incorrect_edges = check_number_incorrect_edges(logit_edges.detach(), edge_features.detach())

        losses = {
            'total_loss': total_loss,
            'node_loss': node_loss,
            'edge_loss': edge_loss,
            'kl_loss': kl_loss,
            'fraction_incorrect': fraction_incorrect,
            'mean_num_incorrect_edges': mean_num_incorrect_edges
        }

        return losses

    def eval(self, dataloader, epoch):
        self.model.eval()

        eval_accumulator = Accumulator()
        eval_start_time = time.perf_counter()
        total_examples = 0

        self.log('Evaluation started at epoch {}', epoch + 1)

        with torch.no_grad():
            for batch in dataloader:
                num_examples = batch['node_features'].shape[0]

                batch = _copy_to_device(batch, self.device)
                losses = self.run_single_batch(batch)

                eval_accumulator.record(_detach_all(losses), weight=num_examples)
                total_examples += num_examples

            if self.dist_config:
                eval_accumulator.reduce()

        if not distributed_utils.is_leader(self.dist_config):
            # Do not return summary information on non-leader process.
            return

        average_loss = _copy_to_device(eval_accumulator.get_average(), 'cpu')
        elapsed = time.perf_counter() - eval_start_time

        self.log('Evaluation done at epoch {}. Done in {}s ({}mol / s)',
                 epoch + 1, elapsed, human_units(total_examples / elapsed))
        self.log('Validation loss: {:.4g}, Node loss: {:.4g}, Edge loss: {:.4g}, KL: {:.4g}',
                 average_loss['total_loss'], average_loss['node_loss'], average_loss['edge_loss'], average_loss['kl_loss'])
        self.log('Mean number of incorrect edges: {:.2f}. Fraction of moleculs incorrect: {:.2%}',
                 average_loss['mean_num_incorrect_edges'], average_loss['fraction_incorrect'])
        self.log('')

        self._summary_scalar('validation_loss/total', average_loss['total_loss'])
        self._summary_scalar('validation_loss/node', average_loss['node_loss'])
        self._summary_scalar('validation_loss/edge', average_loss['edge_loss'])
        self._summary_scalar('validation_loss/kl', average_loss['kl_loss'])
        self._summary_scalar('validation_metrics/fraction_incorrect_molecules', average_loss['fraction_incorrect'])
        self._summary_scalar('validation_metrics/average_incorrect_edges', average_loss['mean_num_incorrect_edges'])
        return average_loss

    def _summary_scalar(self, name, value):
        if self.summary_writer is None:
            return
        self.summary_writer.add_scalar(
            name, value, global_step=self.global_step)

    def record_summary(self, losses, epoch_index, batch_index, batch_size):
        if not distributed_utils.is_leader(self.dist_config):
            # do not record summary on non-leader processes.
            return

        elapsed, num_examples = self.perf_measure.get_counter()

        print("Epoch {}, Batch {}, Iteration {}. {}mol / s".format(
            epoch_index + 1, batch_index + 1, self.global_step,
            human_units(num_examples / elapsed)))
        print("Train loss: {:.3f}, Node loss: {:.3f}, Edge loss: {:.3f}, KL: {:.3f}.".format(
            losses['total_loss'], losses['node_loss'], losses['edge_loss'], losses['kl_loss']))
        print("Mean number incorrect edges: {:.2f}. Fraction of molecules incorrect: {:.2%}.".format(
            losses['mean_num_incorrect_edges'], losses['fraction_incorrect']))

        self._summary_scalar('loss/total', losses['total_loss'])
        self._summary_scalar('loss/node', losses['node_loss'])
        self._summary_scalar('loss/edge', losses['edge_loss'])
        self._summary_scalar('loss/kl', losses['kl_loss'])
        self._summary_scalar('metrics/fraction_incorrect_molecules', losses['fraction_incorrect'])
        self._summary_scalar('metrics/average_incorrect_edges', losses['mean_num_incorrect_edges'])

        # make sure to rescale lr by batch size to obtain lr equivalent to that set by the user
        self._summary_scalar('optim/lr', next(iter(self.optim.param_groups))['lr'] / batch_size * 32)
        self._summary_scalar('molecules_per_second', num_examples / elapsed)

    def state_dict(self):
        return {
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'loss_weights': self.loss_weights,
            'global_step': self.global_step
        }

    def load_state_dict(self, state_dict):
        self.loss_weights = state_dict['loss_weights']
        self.global_step = state_dict['global_step']
        self.model.load_state_dict(state_dict['model'])
        self.optim.load_state_dict(state_dict['optim'])

        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict['scheduler'])
