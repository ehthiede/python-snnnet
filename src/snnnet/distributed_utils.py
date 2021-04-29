"""Utility functions for distributed (multi-gpu) training.
"""

import socket
import typing

import torch
import torch.distributed

from functools import partial


class DistributedTrainingInfo(typing.NamedTuple):
    sync_url: str
    world_size: int
    rank: int
    local_rank: int

    def __bool__(self):
        return self.world_size > 1


def is_leader(config: DistributedTrainingInfo):
    """Tests whether the current process is the leader for distributed training."""
    return config is None or config.rank == 0


def train_boostrap_distributed(parameters, train):
    """Utility to bootstrap a distributed training process.

    This function spawns additional processes if necessary, which each call
    the specified `train` function.

    Parameters
    ----------
    parameters : dict
        Dictionary containing training parameters. This function inspects the keys
        world_size and sync_url to determine the number of processes to spawn.
    train : function
        The training function, takes a single argument representing a dictionary of parameters.
    """
    world_size = parameters.get('world_size', 1)

    if world_size == 1:
        # Single-node training, nothing to do.
        parameters['rank'] = 0
        return train(parameters)

    parameters['rank'] = -1
    from torch.multiprocessing import spawn

    sync_url = parameters.get('sync_url')
    if sync_url is None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        port = s.getsockname()[1]
        s.close()
        sync_url = f'tcp://127.0.0.1:{port}'
        parameters['sync_url'] = sync_url
        print(f'Using URL bootstrap at {sync_url}')

    spawn(partial(train_distributed, train=train), nprocs=parameters['world_size'], args=(parameters,))


def initialize_distributed(config: DistributedTrainingInfo):
    """Initialize the process in a distributed setting."""
    from torch import distributed as dist

    dist.init_process_group(
        'nccl', init_method=config.sync_url,
        world_size=config.world_size, rank=config.rank)

    torch.cuda.set_device(config.local_rank % torch.cuda.device_count())


def get_distributed_config(parameters, local_rank=None):
    world_size = parameters.get('world_size', 1)

    if world_size == 1:
        return DistributedTrainingInfo('', 1, 0, 0)

    sync_url = parameters['sync_url']

    rank = local_rank

    return DistributedTrainingInfo(sync_url, world_size, rank, local_rank)


def train_distributed(local_rank, parameters, train):
    """Set-up the torch distributed framework and call the train function.

    Parameters
    ----------
    local_rank : int
        The rank of given process within the machine.
    parameters : dict
        Dictionary of parameters
    train : function
        Training function
    """
    config = get_distributed_config(parameters, local_rank)
    initialize_distributed(config)
    train(parameters, config)


class SingleDeviceDistributedParallel(torch.nn.parallel.distributed.DistributedDataParallel):
    """This module implements a module similar to `DistributedDataParallel`, but it accepts
    inputs of any shape, and only supports a single device per instance.
    """
    def __init__(self, module, device_id, find_unused_parameters=False):
        super(SingleDeviceDistributedParallel, self).__init__(
            module, [device_id], find_unused_parameters=find_unused_parameters)

    def forward(self, *inputs, **kwargs):
        if self.require_forward_param_sync:
            self._sync_params()

        output = self.module(*inputs, **kwargs)

        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            self.require_forward_param_sync = True

            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(torch.nn.parallel.distributed._find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])

        return output

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.module.load_state_dict(state_dict, strict)
