import argparse
import torch
import itertools
import functools
import numpy as np
from collections import Counter


def move_from_end(x, dim):
    """
    Moves the last dimension to the desired position while preserving ordering of the others.

    Parameters
    ----------
    x : :class:`torch.Tensor`
        Input tensor of shape ... x n
    dim : int
        Dimension to move the last location.

    Returns
    -------
    x : :class:`torch.Tensor`
        Permuted tensor
    """
    N = len(x.shape)
    permute_indices = list(range(N-1))
    permute_indices.insert(dim, N-1)
    return x.permute(permute_indices)


def move_to_end(x, dim):
    """
    Moves a specified dimension to the end.

    """
    N = len(x.shape)
    permute_indices = list(range(N))
    permute_indices.remove(dim)
    permute_indices.append(dim)
    return x.permute(permute_indices)


def channel_product(x, y):
    """
    Takes a product that
    """
    product = x.unsqueeze(-1) * y.unsqueeze(-2)
    return collapse_channels(product)


def collapse_channels(x):
    """
    Collapse two channel inputs into one.
    Must be the last two channels.
    """
    # xs = x.shape
    # x = x.view(xs[:-2] + (xs[-2] * xs[-1],))
    x = torch.flatten(x, -2, -1)
    return x


def get_diag(x, dim1, dim2):
    diag = torch.diagonal(x, dim1=dim1, dim2=dim2)
    return move_from_end(diag, dim1)


def build_idx_3_permutations(orig_shape, pdim1):
    N = len(orig_shape)
    assert(pdim1+2 < N)
    # permute_indices = list(range(N))
    swapping_axes = [pdim1, pdim1+1, pdim1+2]
    full_perms = []
    for sub_perm in itertools.permutations(swapping_axes):
        full_perm = tuple(range(pdim1)) + sub_perm + tuple(range(pdim1+3, N))
        full_perms.append(full_perm)
    return full_perms


def _record_function_decorator(func, name=None):
    if name is None:
        name = func.__name__

    def wrap(*args, **kwargs):
        with torch.autograd.profiler.record_function(name):
            return func(*args, **kwargs)

    return wrap


def record_function(name=None):
    """Records the function as a block in the pytorch autograd profiler.

    This decorator ensures that the decorated function can be seen as a block in the pytorch autograd profiler.
    By default, the name used is that of the function, although a custom name can be given.

    Parameters
    ----------
    name : str, optional
        The name to give the block. If `None`, attempts to extract the name of the decorated function.
    """
    return functools.partial(_record_function_decorator, name=name)


def human_units(value, base=1000):
    """Format number using human-scaled units, useful for large numbers (e.g. bytes)."""
    for unit in ['', 'K', 'M', 'G', 'T']:
        if value < base:
            break
        value /= base
    else:
        unit = 'P'

    return "{:.4g} {}".format(value, unit)


def str2bool(v):
    """Utility function to parse boolean argument values using `argparse`.
    """
    if v is None:
        return None

    if isinstance(v, bool):
        return v

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

    raise argparse.ArgumentTypeError('Boolean value expected')


class LengthSampler:
    """
    Reads in a dataset and returns samples with the associated positional embeddings
    """
    def __init__(self, train_dataset, batch_size=1):
        lengths = [torch.max(mol['positional_features']).item() for mol in train_dataset]
        print(train_dataset[0]['positional_features'].shape, 'pos feat shape')
        self.num_atoms = train_dataset[0]['positional_features'].shape[0]
        self.max_length = np.max(lengths)
        ctr = Counter(lengths)
        self.freqs = torch.zeros(self.max_length)
        print(self.freqs)
        for key, val in ctr.items():
            self.freqs[key-1] = val
        self.freqs /= torch.sum(self.freqs)
        self.batch_size = batch_size

    def __call__(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        lengths = np.random.choice(np.arange(1, self.max_length+1), size=batch_size, p=self.freqs)
        pos_feats = torch.zeros(batch_size, self.num_atoms)
        for i, length in enumerate(lengths):
            pos_feats[i, :length] = torch.arange(length)+1
        return pos_feats.long()
