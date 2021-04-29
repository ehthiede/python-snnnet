"""Main training script for molecule VAE model."""

import argparse
import bisect
import datetime
import functools
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.utils.tensorboard

from snnnet import arguments, distributed_utils, utils
from snnnet.distributed_utils import train_boostrap_distributed
from snnnet.molecule import training, molecule_vae, molgraph_dataset


def _warmup_decay_lr(epoch, warmup_epochs=0, decay_epochs=None):
    if decay_epochs is None:
        decay_epochs = []

    warmup_factor = min(epoch + 1, warmup_epochs + 1) / (warmup_epochs + 1)
    decay_factor = 0.1 ** bisect.bisect_left(decay_epochs, epoch)

    return warmup_factor * decay_factor


def _load_dataset(dataset_path):
    print('Loading dataset from {}'.format(dataset_path))
    mols = molgraph_dataset.MoleculeData(np.load(dataset_path))
    mols.share_memory_()
    return molgraph_dataset.MolGraphDataset(mols)


def _save_model(save_path, harness, args, epoch):
    torch.save({
        'harness': harness.state_dict(),
        'training_args': args,
        'epoch': epoch},
        save_path,
        pickle_protocol=pickle.HIGHEST_PROTOCOL)


def train_with_args(args, dist_config: distributed_utils.DistributedTrainingInfo=None):
    args = ARGUMENT_SET.fill_dictionary_with_defaults(args)
    args = _set_up_file_structure(args)

    torch.manual_seed(args['seed'])

    total_batch_size = args['batch_size']
    worker_batch_size = total_batch_size

    if dist_config:
        # each separate process is only responsible for a fraction of total batch size
        worker_batch_size //= dist_config.world_size

    with open(os.path.join(args['directory'], 'settings.json'), 'w') as f:
        json.dump(args, f, indent=4, sort_keys=True)

    train_dataset = _load_dataset(args['train_dataset'])
    valid_dataset = _load_dataset(args['valid_dataset'])

    if dist_config:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, seed=args['seed'])
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dataset, shuffle=False)
    else:
        train_sampler=None
        valid_sampler = None

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, worker_batch_size, pin_memory=True, num_workers=8,
        shuffle=False if train_sampler is not None else True)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, 2 * worker_batch_size, pin_memory=True,
        sampler=valid_sampler)
    print('Done loading datasets')

    model = molecule_vae.make_molecule_vae(
        args['encoder_channels'], args['decoder_channels'],
        args['latent_channels'], train_dataset.node_feature_cardinality,
        latent_transform_depth=args['latent_transform_depth'],
        node_embedding_dimension=args['node_embedding_dimension'],
        positional_embedding_dimension=args['positional_embedding_dimension'],
        batch_norm=args['batch_norm'],
        expand_type=args['expand_type'],
        architecture=args['architecture'],
        max_size=train_dataset.pad_size)

    if dist_config:
        device_count = torch.cuda.device_count()
        if dist_config.local_rank >= device_count:
            print('Warning: spawned more processes than GPUs. Multiple workers will share a GPU.\n'
                  'This is probably not what you want except for testing purposes.')

        gpu_id = dist_config.local_rank % device_count

        print('Creating model on GPU {0}'.format(gpu_id))
        device = torch.device('cuda', gpu_id)
        model = distributed_utils.SingleDeviceDistributedParallel(
            model.to(device), gpu_id)
    else:
        device = torch.device('cuda')
        model = model.to(device)

    print('Model done building')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args['learning_rate'] * total_batch_size / 32, weight_decay=args['weight_decay'])

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, functools.partial(_warmup_decay_lr, warmup_epochs=5, decay_epochs=[20, 40]))

    pos_weight = torch.tensor(args['pos_weight']).to(device)

    if distributed_utils.is_leader(dist_config):
        summary_writer = torch.utils.tensorboard.SummaryWriter(log_dir=args['directory'], flush_secs=60)
    else:
        summary_writer = None

    harness = training.Harness(
        args['directory'], model, optimizer,
        loss_weights={'pos': pos_weight, 'edge': args['edge_weight'], 'node': args['node_weight'], 'kl': args['kl_weight']},
        scheduler=scheduler, device=device, summary=summary_writer,
        train_sampler=train_sampler, dist_config=dist_config)

    train_start = time.perf_counter()
    print('Starting training.')

    final_eval = None

    for i in range(args['epochs']):
        harness.train_epoch(train_dataloader, i)
        final_eval = harness.eval(valid_dataloader, i)

        if (i + 1) % args['save_frequency'] == 0 and distributed_utils.is_leader(dist_config):
            save_path = os.path.join(args['directory'], 'train_state_{}.pth'.format(i + 1))
            print('Saving current state to: {}'.format(save_path))
            _save_model(save_path, harness, args, i)

        save_path = os.path.join(args['directory'], 'current_state.pth'.format(i + 1))
        _save_model(save_path, harness, args, i)

    if final_eval is not None:
        print('Writing hparam summary at final epoch.')
        summary_writer.add_hparams(
            _get_hparams_from_args(args), {
                'final_metric/total_loss': final_eval['total_loss'],
                'final_metric/node_loss': final_eval['node_loss'],
                'final_metric/edge_loss': final_eval['edge_loss'],
                'final_metric/kl_loss': final_eval['kl_loss'],
                'final_metric/fraction_incorrect_molecules': final_eval['fraction_incorrect'],
                'final_metric/average_incorrect_edges': final_eval['mean_num_incorrect_edges']
            })

    train_elapsed = time.perf_counter() - train_start
    print('Training for {} epochs done in {}'.format(args['epochs'], datetime.timedelta(seconds=int(train_elapsed))))


HPARAM_KEYS = (
    'batch_size',
    'learning_rate',
    'pos_weight',
    'node_weight',
    'edge_weight',
    'kl_weight',
    'latent_channels',
    'weight_decay',
    'latent_transform_depth',
    'node_embedding_dimension',
    'batch_norm',
)


def _make_argument_set():
    argset = arguments.ArgumentSet()
    argset.add_argument('directory', 'train', help='The training directory, used to store all training information and logs')
    argset.add_argument('train_dataset', 'data/qm9_train.npz', help='Path to the training dataset')
    argset.add_argument('valid_dataset', 'data/qm9_valid.npz', help='Path to the validation dataset')
    argset.add_argument('batch_size', 256, type=int, help='Batch size for training')
    argset.add_argument('learning_rate', 1e-3, type=float, help='Learning rate for training (note: it will be scaled by the batch size)')
    argset.add_argument('epochs', 60, type=int, help='Total number of epochs for which to train the model')
    argset.add_argument('pos_weight', 3.0, type=float, help='Weight of the present edges compared to absent edges')
    argset.add_argument('node_weight', 1.0, type=float, help='Weight of the node recovery loss')
    argset.add_argument('edge_weight', 100.0, type=float, help='Weight of the edge recovery loss')
    argset.add_argument('kl_weight', 1.0, type=float, help='Weight of the KL-divergence loss')
    argset.add_argument('architecture', 'o3', type=str, choices=list(molecule_vae.CORE_ARCHITECTURE.keys()),
                        help='Architecture family for the encoder and decoder of the VAE.')
    argset.add_argument('encoder_channels', [32, 32], type=int, nargs='+', help='Number of channels to use at each encoder layer')
    argset.add_argument('decoder_channels', [32, 32], type=int, nargs='+', help='Number of channels to use at each decoder layer')
    argset.add_argument('latent_channels', 4, type=int, help='Number of channels to use in the latent dimension')
    argset.add_argument('save_frequency', 10, type=int, help='Frequency with which to save models to file')
    argset.add_argument('latent_transform_depth', 1, type=int, help='Depth of the fully-connected transforms before latent layer')
    argset.add_argument('weight_decay', 1e-2, type=float, help='Strength of the weight decay parameter')
    argset.add_argument('node_embedding_dimension', 0, type=int, help='Number of dimensions to use for node feature embedding. 0 disables embedding and uses one-hot encoding directly.')
    argset.add_argument('positional_embedding_dimension', 0, type=int, help='Dimension of learned positional embedding (for sequence position in SMILES). 0 disables positional embeddinsg.')
    argset.add_argument('batch_norm', True, type=bool, help='Whether to use batch normalization in layers')
    argset.add_argument('expand_type', 'direct', choices=['linear', 'direct', 'individual'], help='Type of expansion to lift tensor dimension')
    argset.add_argument('seed', 2, type=int, help='Random seed controlling all random number generation in training')
    argset.add_argument('world_size', 1, type=int, help='Number of GPUs to use.')
    return argset

ARGUMENT_SET = _make_argument_set()
ARGUMENT_SET.__doc__ = "The set of all arguments used for training molecule VAEs."


def _get_hparams_from_args(args):
    """Obtains a dictionary which is compatible with tensorboard's HPARAMs describing the training hyper-parameters."""
    hparams = {k: args[k] for k in HPARAM_KEYS}
    hparams['encoder_width'] = args['encoder_channels'][0]
    hparams['encoder_depth'] = len(args['encoder_channels'])
    hparams['decoder_width'] = args['decoder_channels'][0]
    hparams['decoder_depth'] = len(args['decoder_channels'])
    return hparams


def _set_up_file_structure(args):
    """
    Converts paths to absolute file paths and initializes working directory
    (if it doesn't already exist).
    """
    print(args['directory'])
    os.makedirs(args['directory'], exist_ok=True)
    args['directory'] = os.path.abspath(args['directory'])
    args['train_dataset'] = os.path.abspath(args['train_dataset'])
    args['valid_dataset'] = os.path.abspath(args['valid_dataset'])
    return args


def main():
    args = ARGUMENT_SET.parse_args()
    distributed_utils.train_boostrap_distributed(args, train_with_args)


if __name__ == '__main__':
    main()
