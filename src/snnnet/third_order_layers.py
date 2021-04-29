"""This module implements the basic building block for permutation equivariant objects."""

import math
import numpy as np

import torch
import torch.nn as nn

from snnnet.utils import move_to_end, get_diag, build_idx_3_permutations, record_function
from snnnet import normalizations as sn_norm


class _SnConvBase(nn.Module):
    """Shared functionality for 3rd-order permutation-equivariant convolutions."""
    def __init__(self, in_channels, out_channels, pdim1=1, in_node_channels=0, diagonal_free=False, init_method=None, inner_bias=True):
        super().__init__()

        if pdim1 != 1:
            raise NotImplementedError('convolution only implemented for NHWDC format (pdim == 1).')

        self.pdim1 = pdim1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_node_channels = in_node_channels
        self.diagonal_free = diagonal_free
        self._init_method = init_method
        self.inner_bias = inner_bias

    def _make_linear(self, in_channels, out_channels):
        """Makes a linear layer with common configuration for the convolution.

        This makes a new linear layer and initializes the layer according to
        the specified weight initialization method.
        """
        layer = nn.Linear(in_channels, out_channels, bias=self.inner_bias)
        if self._init_method is not None:
            self._init_method(layer.weight)
        return layer

    def _build_first_order_linears(self):
        """
        Builds linear layers for entries going into the diagonal slices of the tensor
        """
        in_c = self.in_channels
        out_c = self.out_channels
        self.one_to_one_linear = self._make_linear(10 * in_c + self.in_node_channels, out_c)
        self.zero_to_one_linear = self._make_linear(5 * in_c, out_c)

    @record_function()
    def mix_first_order(self, a1, a0):
        """
        Mixes the core vector of the third order tensor.

        Parameters
        ----------
        a1 : Tensor,
            First order contractions
        a0 : Tensor,
            Zeroth order contractions
        """
        pdim1 = self.pdim1
        all_first_order = self.one_to_one_linear(a1)
        all_first_order += self.zero_to_one_linear(a0).unsqueeze(pdim1)

        if self.diagonal_free:
            all_first_order *= 1 / math.sqrt(2)

        return all_first_order

    def _mix_second_order_helper(self, a2_w_t, a1, a0, linears):
        """Mixes all of the diagonal slices.

        Parameters
        ----------
        a2 : Tensor,
            Second order contractions, concatenate with its transpose
        a1 : Tensor,
            First order contractions
        a0 : Tensor,
            Zeroth order contractions
        """
        pdim1 = self.pdim1
        two_to_two, one_to_two, one_to_two_t, zero_to_two = linears

        second_order_mix = two_to_two(a2_w_t)
        second_order_mix += one_to_two(a1).unsqueeze(pdim1+1)
        second_order_mix += one_to_two_t(a1).unsqueeze(pdim1)
        second_order_mix += zero_to_two(a0).unsqueeze(pdim1).unsqueeze(pdim1+1)

        if self.diagonal_free:
            second_order_mix *= 0.5

        return second_order_mix


class SnConv3(_SnConvBase):
    """
    Convolutional layer for 3rd order permutation-equivariant objects

    Parameters
    ----------
    in_channels : int
        The size of the channel index of the incoming tensor
    out_channels : int
        The size of the channel index of the outgoing tensor
    in_node_channels : int
        Number of input node channels.  Default is zero.
    pdim1 : int
        First dimension on which the permutation group acts.
        Second dimension on which the group acts must be pdim1+1
    diagonal_free : bool, optional
        If `True`, indicates that the parametrization should be diagonal-free and switches
        normalization to preserve variance.
    inner_bias : bool, optional
        Whether to use bias in the inner linear transformations which make up the transformation.
    """

    def __init__(self, in_channels, out_channels, pdim1=1, contraction_norm=None,
                 mixture_norm=None, in_node_channels=0,
                 diagonal_free=False, init_method=None, inner_bias=True):
        super().__init__(in_channels, out_channels, pdim1, in_node_channels, diagonal_free, init_method, inner_bias)

        self._build_third_order_linears()
        self._build_second_order_linears()
        self._build_first_order_linears()
        self._build_contraction_normalization(contraction_norm)
        self._build_mixture_normalization(mixture_norm)

    def _make_linear(self, in_channels, out_channels):
        """Makes a linear layer with common configuration for the convolution.

        This makes a new linear layer and initializes the layer according to
        the specified weight initialization method.
        """
        layer = nn.Linear(in_channels, out_channels, bias=self.inner_bias)
        if self._init_method is not None:
            self._init_method(layer.weight)
        return layer

    def _build_third_order_linears(self, init_method=None):
        """
        Builds linear layers for entries going into the entire third order tensor.
        """
        in_c = self.in_channels
        in_node_c = self.in_node_channels
        out_c = self.out_channels
        self.three_to_three_linear = self._make_linear(6 * in_c, out_c)

        self.two_to_three_linear = nn.ModuleList()
        for i in range(3):
            o2_module_pair = self._make_linear(12 * in_c, out_c)
            self.two_to_three_linear.append(o2_module_pair)

        self.one_to_three_linear = nn.ModuleList([self._make_linear(10 * in_c + in_node_c, out_c) for i in range(3)])
        self.zero_to_three_linear = self._make_linear(5 * in_c, out_c)

    def _build_second_order_linears(self):
        """
        Builds linear layers for entries going into the 2d diagonal slices of the tensor
        """
        in_c = self.in_channels
        out_c = self.out_channels
        in_node_c = self.in_node_channels
        self.second_order_linears = nn.ModuleList()
        for i in range(3):
            slice_layers = nn.ModuleList()
            slice_layers.append(self._make_linear(12 * in_c, out_c))
            slice_layers.append(self._make_linear(10 * in_c + in_node_c, out_c))
            slice_layers.append(self._make_linear(10 * in_c + in_node_c, out_c))
            slice_layers.append(self._make_linear(5 * in_c, out_c))
            self.second_order_linears.append(slice_layers)

    def _build_contraction_normalization(self, contraction_norm):
        in_c = self.in_channels
        in_node_c = self.in_node_channels
        self.cnorm_layers = nn.ModuleList()
        if contraction_norm is None:
            for i in range(4):
                self.cnorm_layers.append(nn.Identity())
        elif contraction_norm.lower() == 'batch':
            self.cnorm_layers.append(sn_norm.SnBatchNorm1d(5 * in_c))  # O0 Tensors
            self.cnorm_layers.append(sn_norm.SnBatchNorm1d(10 * in_c + in_node_c))  # O1 Tensors
            self.cnorm_layers.append(sn_norm.SnBatchNorm2d(6 * in_c))  # O2 Tensors
            self.cnorm_layers.append(sn_norm.SnBatchNorm3d(in_c))  # O3 Tensors
        elif contraction_norm.lower() == 'instance':
            self.cnorm_layers.append(nn.Identity())  # No 0-order instance norm possible.
            self.cnorm_layers.append(sn_norm.SnInstanceNorm1d(10 * in_c + in_node_c))  # O1 Tensors
            self.cnorm_layers.append(sn_norm.SnInstanceNorm2d(6 * in_c))  # O2 Tensors
            self.cnorm_layers.append(sn_norm.SnInstanceNorm3d(in_c))  # O3 Tensors
        else:
            raise ValueError('Name for the normalization for the initial contractions not recognized')

    def _build_mixture_normalization(self, mixture_norm):
        out_c = self.out_channels
        self.mnorm_layers = nn.ModuleList()
        if mixture_norm is None:
            self.mnorm_layers.append(nn.Identity())  # O1 Tensors
            self.mnorm_layers.append(nn.ModuleList([nn.Identity() for i in range(3)]))
            self.mnorm_layers.append(nn.Identity())  # O3 Tensors
        elif mixture_norm.lower() == 'batch':
            self.mnorm_layers.append(sn_norm.SnBatchNorm1d(out_c))  # O1 Tensors
            o2_mnorm_layers = nn.ModuleList([sn_norm.SnBatchNorm2d(out_c) for i in range(3)])
            self.mnorm_layers.append(o2_mnorm_layers)
            self.mnorm_layers.append(sn_norm.SnBatchNorm3d(out_c))  # O3 Tensors
        elif mixture_norm.lower() == 'instance':
            self.mnorm_layers.append(sn_norm.SnInstanceNorm1d(out_c))  # O1 Tensors
            o2_mnorm_layers = nn.ModuleList([sn_norm.SnInstanceNorm2d(out_c) for i in range(3)])
            self.mnorm_layers.append(o2_mnorm_layers)
            self.mnorm_layers.append(sn_norm.SnInstanceNorm3d(out_c))  # O3 Tensors
        else:
            raise ValueError('Name for the normalization for the mixed tensors not recognized')

    def normalize_contractions(self, a3, a2, a1, a0):
        a0 = self.cnorm_layers[0](a0)
        a1 = self.cnorm_layers[1](a1)
        a2 = self.cnorm_layers[2](a2)
        a3 = self.cnorm_layers[3](a3)
        return a3, a2, a1, a0

    def normalize_mixed_tensors(self, mixed_full, mixed_slices, mixed_diag):
        mixed_diag = self.mnorm_layers[0](mixed_diag)
        mixed_slices = []
        for a1_i, layer_i in zip(mixed_slices, self.mnorm_layers[1]):
            mixed_slices.append(layer_i(a1_i))
        mixed_full = self.mnorm_layers[2](mixed_full)
        return mixed_full, mixed_slices, mixed_diag

    @record_function()
    def mix_third_order(self, a3, a2, a1, a0):
        """
        Mixes all of the  contractions into the bulk matrix.

        Parameters
        ----------
        a3 : Tensor,
            Third order contractions
        a2 : Tensor,
            Second order contractions
        a1 : Tensor,
            First order contractions
        a0 : Tensor,
            Zeroth order contractions
        """
        pdim1 = self.pdim1

        third_order_mix = _third_order_permutation_matmul_reference(self.three_to_three_linear, a3)

        a2_w_transpose = torch.cat([a2, torch.transpose(a2, pdim1, pdim1+1)], dim=-1)
        for i, o2_module_pair in enumerate(self.two_to_three_linear):
            from_o2_mixed = o2_module_pair(a2_w_transpose)
            third_order_mix += from_o2_mixed.unsqueeze(pdim1 + i)

        unsqueeze_dims = [(pdim1, pdim1 + 1), (pdim1, pdim1+2), (pdim1+1, pdim1+2)]
        for i, usd in enumerate(unsqueeze_dims):
            from_o1_mixed = self.one_to_three_linear[i](a1)
            third_order_mix += from_o1_mixed.unsqueeze(usd[0]).unsqueeze(usd[1])

        from_o0_mixed = self.zero_to_three_linear(a0)
        third_order_mix += from_o0_mixed.unsqueeze(pdim1).unsqueeze(pdim1+1).unsqueeze(pdim1+2)

        if self.diagonal_free:
            third_order_mix *= 1 / math.sqrt(1 + 3 + 3 + 1)

        return third_order_mix

    @record_function()
    def mix_second_order(self, a2, a1, a0):
        """
        Mixes all of the diagonal slices.

        Parameters
        ----------
        a2 : Tensor,
            Second order contractions
        a1 : Tensor,
            First order contractions
        a0 : Tensor,
            Zeroth order contractions
        """
        pdim1 = self.pdim1
        slice_mixes = []

        with torch.autograd.profiler.record_function('assemble_permutations_2'):
            a2_w_transpose = torch.cat([a2, torch.transpose(a2, pdim1, pdim1+1)], dim=-1)

        for i, slice_layers in enumerate(self.second_order_linears):
            second_order_mix = self._mix_second_order_helper(a2_w_transpose, a1, a0, slice_layers)
            slice_mixes.append(second_order_mix)

        return slice_mixes

    @record_function('snconv3')
    def forward(self, x, x_node=None, mask=None):
        """
        Runs a 2d convolution on the input data.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Input to the convolution, of shape B x n x n x C_in
            where B is the batch size, C_in is the number of channels,
            and n is the size of the permutation
        x_node : :class:`torch.Tensor` or None, optional
            Node features to input, of shape B x n x C_in
        mask : :class:`torch.ByteTensor`
            Mask used to account for graphs of unequal size.
            True if a valid atom, False otherwise
        """
        pdim1 = self.pdim1
        pdim2 = pdim1 + 1
        pdim3 = pdim1 + 2

        a3 = x

        with torch.autograd.profiler.record_function('compute_contractions'):
            a2, a1, a0 = _get_third_order_contractions(x, dim1=pdim1, reduction='sqrt' if self.diagonal_free else 'mean')
            if x_node is not None:
                a1 = torch.cat([a1, x_node], dim=-1)
            self.normalize_contractions(a3, a2, a1, a0)

        mixed_full = self.mix_third_order(a3, a2, a1, a0)
        mixed_slices = self.mix_second_order(a2, a1, a0)
        mixed_diag = self.mix_first_order(a1, a0)

        self.normalize_mixed_tensors(mixed_full, mixed_slices, mixed_diag)

        with torch.autograd.profiler.record_function('expand_mix'):
            _expand_mix(mixed_full, mixed_slices, mixed_diag, pdim1, pdim2, pdim3, diagonal_free=self.diagonal_free)

        return mixed_full


class SnConv3to2(_SnConvBase):
    def __init__(self, in_channels, out_channels, pdim1=1, contraction_norm=None, mixture_norm=None,
                 diagonal_free=False, init_method=None, inner_bias=True):
        super().__init__(in_channels, out_channels, pdim1, 0, diagonal_free, init_method, inner_bias)

        self._build_second_order_linears()
        self._build_first_order_linears()
        self._build_contraction_normalization(contraction_norm)
        self._build_mixture_normalization(mixture_norm)

    def _build_second_order_linears(self):
        """
        Builds linear layers for entries going into the 2d diagonal slices of the tensor
        """
        in_c = self.in_channels
        out_c = self.out_channels
        self.second_order_linears = nn.ModuleList()
        self.second_order_linears.append(self._make_linear(12 * in_c, out_c))
        self.second_order_linears.append(self._make_linear(10 * in_c, out_c))
        self.second_order_linears.append(self._make_linear(10 * in_c, out_c))
        self.second_order_linears.append(self._make_linear(5 * in_c, out_c))

    def _build_contraction_normalization(self, contraction_norm):
        in_c = self.in_channels
        self.cnorm_layers = nn.ModuleList()
        if contraction_norm is None:
            for i in range(3):
                self.cnorm_layers.append(nn.Identity())
        elif contraction_norm.lower() == 'batch':
            self.cnorm_layers.append(sn_norm.SnBatchNorm1d(5 * in_c))  # O0 Tensors
            self.cnorm_layers.append(sn_norm.SnBatchNorm1d(10 * in_c))  # O1 Tensors
            self.cnorm_layers.append(sn_norm.SnBatchNorm2d(6 * in_c))  # O2 Tensors
        elif contraction_norm.lower() == 'instance':
            self.cnorm_layers.append(nn.Identity())  # No 0-order instance norm possible.
            self.cnorm_layers.append(sn_norm.SnInstanceNorm1d(10 * in_c))  # O1 Tensors
            self.cnorm_layers.append(sn_norm.SnInstanceNorm2d(6 * in_c))  # O2 Tensors
        else:
            raise ValueError('Name for the normalization for the initial contractions not recognized')

    def _build_mixture_normalization(self, mixture_norm):
        out_c = self.out_channels
        self.mnorm_layers = nn.ModuleList()
        if mixture_norm is None:
            self.mnorm_layers.append(nn.Identity())  # O1 Tensors
            self.mnorm_layers.append(nn.Identity())  # O2 Tensors
        elif mixture_norm.lower() == 'batch':
            self.mnorm_layers.append(sn_norm.SnBatchNorm1d(out_c))  # O1 Tensors
            self.mnorm_layers.append(sn_norm.SnBatchNorm2d(out_c))  # O2 Tensors
        elif mixture_norm.lower() == 'instance':
            self.mnorm_layers.append(sn_norm.SnInstanceNorm1d(out_c))  # O1 Tensors
            self.mnorm_layers.append(sn_norm.SnInstanceNorm2d(out_c))  # O2 Tensors
        else:
            raise ValueError('Name for the normalization for the mixed tensors not recognized')

    def normalize_contractions(self, a2, a1, a0):
        a0 = self.cnorm_layers[0](a0)
        a1 = self.cnorm_layers[1](a1)
        a2 = self.cnorm_layers[2](a2)
        return a2, a1, a0

    def normalize_mixed_tensors(self, mixed_full, mixed_diag):
        mixed_diag = self.mnorm_layers[0](mixed_diag)
        mixed_full = self.mnorm_layers[1](mixed_full)
        return mixed_full, mixed_diag

    def mix_second_order(self, a2, a1, a0):
        """
        Mixes all of the diagonal slices.

        Parameters
        ----------
        a2 : Tensor,
            Second order contractions
        a1 : Tensor,
            First order contractions
        a0 : Tensor,
            Zeroth order contractions
        """
        pdim1 = self.pdim1
        two_to_two, one_to_two, one_to_two_t, zero_to_two = self.second_order_linears

        a2_w_transpose = torch.cat([a2, torch.transpose(a2, pdim1, pdim1+1)], dim=-1)
        return self._mix_second_order_helper(a2_w_transpose, a1, a0, self.second_order_linears)

    def forward(self, x, mask=None):
        """
        Runs a 2d convolution on the input data.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Input to the convolution, of shape B x n x n x C_in
            where B is the batch size, C_in is the number of channels,
            and n is the size of the permutation
        x_node : :class:`torch.Tensor` or None, optional
            Node features to input, of shape B x n x C_in
        mask : :class:`torch.ByteTensor`
            Mask used to account for graphs of unequal size.
            True if a valid atom, False otherwise
        """
        pdim1 = self.pdim1
        pdim2 = pdim1 + 1

        a2, a1, a0 = _get_third_order_contractions(x, dim1=pdim1, reduction='sqrt' if self.diagonal_free else 'mean')
        a2, a1, a0 = self.normalize_contractions(a2, a1, a0)

        mixed_full = self.mix_second_order(a2, a1, a0)
        mixed_diag = self.mix_first_order(a1, a0)
        mixed_full, mixed_diag = self.normalize_mixed_tensors(mixed_full, mixed_diag)

        mixed_diag = move_to_end(mixed_diag, dim=pdim1)

        mixed_full_diag = mixed_full.diagonal(dim1=pdim1, dim2=pdim2)

        if self.diagonal_free:
            mixed_full_diag[...] = mixed_diag
        else:
            mixed_full_diag.add_(mixed_diag)

        return mixed_full


class SnConv3to1(_SnConvBase):
    def __init__(self, in_channels, out_channels, pdim1=1, contraction_norm=None, mixture_norm=None,
                 diagonal_free=False, init_method=None, inner_bias=True):
        super().__init__(in_channels, out_channels, pdim1, 0, diagonal_free, init_method, inner_bias)

        self._build_first_order_linears()
        self._build_contraction_normalization(contraction_norm)
        self._build_mixture_normalization(mixture_norm)

    def _build_contraction_normalization(self, contraction_norm):
        in_c = self.in_channels
        self.cnorm_layers = nn.ModuleList()
        if contraction_norm is None:
            for i in range(2):
                self.cnorm_layers.append(nn.Identity())
        elif contraction_norm.lower() == 'batch':
            self.cnorm_layers.append(sn_norm.SnBatchNorm1d(5 * in_c))  # O0 Tensors
            self.cnorm_layers.append(sn_norm.SnBatchNorm1d(10 * in_c))  # O1 Tensors
        elif contraction_norm.lower() == 'instance':
            self.cnorm_layers.append(nn.Identity())  # No 0-order instance norm possible.
            self.cnorm_layers.append(sn_norm.SnInstanceNorm1d(10 * in_c))  # O1 Tensors
        else:
            raise ValueError('Name for the normalization for the initial contractions not recognized')

    def _build_mixture_normalization(self, mixture_norm):
        out_c = self.out_channels
        self.mnorm_layers = nn.ModuleList()
        if mixture_norm is None:
            self.mnorm_layers.append(nn.Identity())  # O1 Tensors
        elif mixture_norm.lower() == 'batch':
            self.mnorm_layers.append(sn_norm.SnBatchNorm1d(out_c))  # O1 Tensors
        elif mixture_norm.lower() == 'instance':
            self.mnorm_layers.append(sn_norm.SnInstanceNorm1d(out_c))  # O1 Tensors
        else:
            raise ValueError('Name for the normalization for the mixed tensors not recognized')

    def normalize_contractions(self, a1, a0):
        a0 = self.cnorm_layers[0](a0)
        a1 = self.cnorm_layers[1](a1)
        return a1, a0

    def normalize_mixed_tensors(self, x):
        x = self.mnorm_layers[0](x)
        return x

    @record_function('snconv3to1')
    def forward(self, x, mask=None):
        """
        Runs a 2d convolution on the input data.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Input to the convolution, of shape B x n x n x C_in
            where B is the batch size, C_in is the number of channels,
            and n is the size of the permutation
        x_node : :class:`torch.Tensor` or None, optional
            Node features to input, of shape B x n x C_in
        mask : :class:`torch.ByteTensor`
            Mask used to account for graphs of unequal size.
            True if a valid atom, False otherwise
        """
        __, a1, a0 = _get_third_order_contractions(x, dim1=self.pdim1, reduction='sqrt' if self.diagonal_free else 'mean')
        self.normalize_contractions(a1, a0)
        x = self.mix_first_order(a1, a0)
        x = self.normalize_mixed_tensors(x)
        return x


def expand_2_to_3(x, mode='subdivide', pdim1=1):
    nc = x.shape[-1]
    N = x.shape[pdim1]
    x_ones = x.unsqueeze(-2).repeat([1, 1, 1, N, 1])
    out_list = []
    if mode == 'subdivide':
        subc = (nc // 3)
        # assert(3 * subc == nc)
        x1 = x[..., :subc]
        x2 = x[..., subc:2*subc]
        x3 = x[..., 2*subc:]
        x_squared = torch.einsum('bijc,bjkd->bijkcd', x1, x2)
        x_squared = x_squared.flatten(start_dim=-2)
        x_cubed = torch.einsum('bijkc,bikd->bijkcd', x_squared, x3)
        x_cubed = x_cubed.flatten(start_dim=-2)
        out_list += [x_ones, x_squared, x_cubed]
    elif mode == 'all':
        x_squared = torch.einsum('bijc,bjkd->bijkcd', x, x)
        x_squared = x_squared.flatten(start_dim=-2)
        x_cubed = torch.einsum('bijkc,bikd->bijkcd', x_squared, x)
        x_cubed = x_cubed.flatten(start_dim=-2)
        out_list += [x_ones, x_squared, x_cubed]
    elif mode == 'individual':
        x_squared = torch.einsum('bijc,bjkc->bijkc', x, x)
        x_cubed = torch.einsum('bijkc,bikc->bijkc', x_squared, x)
        out_list += [x_ones, x_squared, x_cubed]
    elif mode == 'linear':
        x_lift = move_to_end(x, dim=pdim1+1)
        x_lift = torch.diag_embed(x_lift, dim1=pdim1+1, dim2=pdim1+2)
        out_list.append(x_lift)

    # x_out = torch.cat([x_ones, x_squared, x_cubed], dim=-1)
    x_out = torch.cat(out_list, dim=-1)
    return x_out


def expand_1_to_2(x, mode='individual', pdim1=1):
    if mode == 'individual':
        x_t = x.unsqueeze(pdim1+1)
        x_sq = x.unsqueeze(pdim1)
        x_sq = x_sq * x_t
    elif mode == 'all':
        x_t = x.unsqueeze(pdim1+1).unsqueeze(-1)
        x_sq = x.unsqueeze(pdim1).unsqueeze(-2)
        x_sq = x_sq * x_t
        new_shape = x_sq.shape[:-2] + (x_sq.shape[-1] * x_sq.shape[-2],)
        x_sq = x_sq.view(new_shape)
    else:
        raise ValueError('outer type not recognized')
    x_diag = move_to_end(x, pdim1)
    x_diag = torch.diag_embed(x_diag, dim1=pdim1, dim2=pdim1+1)
    return torch.cat([x_diag, x_sq], dim=-1)


def expand_1_to_3(x, mode='direct', pdim1=1):
    # Third order tensor product
    if mode == 'direct':
        x_t = x.unsqueeze(pdim1+1)
        x_sq = x.unsqueeze(pdim1)
        x_sq = x_sq * x_t
        x_t2 = x.unsqueeze(pdim1).unsqueeze(pdim1+1)
        x_cube = x_sq.unsqueeze(pdim1+2) * x_t2

        # Second order tensor product
        x_sq_on_diag = move_to_end(x_sq, pdim1+1)
        expanded_slice = torch.diag_embed(x_sq_on_diag, dim1=pdim1+1, dim2=pdim1+2)

        x_diag = move_to_end(x, pdim1)
        expanded_diag = torch.diag_embed(x_diag, dim1=-1, dim2=pdim1)
        expanded_diag = torch.diag_embed(expanded_diag, dim1=pdim1+1, dim2=pdim1+2)
        x = torch.cat([x_cube, expanded_slice, expanded_diag], dim=-1)
    elif mode == 'linear':
        x = move_to_end(x, dim=pdim1)
        x = torch.diag_embed(x, dim1=-1, dim2=pdim1)
        x = torch.diag_embed(x, dim1=pdim1+1, dim2=pdim1+2)
    else:
        raise ValueError('outer type not recognized')
    return x


def _compute_second_order_contractions_reference(x, dims):
    dim_pairs = [(dims[0], dims[1]), (dims[0], dims[2]), (dims[1], dims[2])]

    second_order_sums = [torch.mean(x, dim=dim) for dim in dims]
    second_order_diags = [get_diag(x, dim1=dima, dim2=dimb) for (dima, dimb) in dim_pairs]
    all_second_order = second_order_sums + second_order_diags
    stacked_second_order = torch.cat(all_second_order, dim=-1)

    return second_order_diags, stacked_second_order


def _generic_reduction(x, dim, out=None, reduction='mean'):
    """Helper function to compute a generic sum-like reduction.

    This function generalizes mean, sum, and sqrt-normalized sums.
    """

    if reduction == 'mean':
        return torch.mean(x, dim=dim, out=out)
    elif reduction == 'sum':
        return torch.sum(x, dim=dim, out=out)
    elif reduction == 'sqrt':
        factor = 1 / math.sqrt(np.prod([x.shape[i] for i in np.atleast_1d(dim)]))
        if out is None:
            return factor * torch.sum(x, dim=dim)
        else:
            torch.sum(x, dim=dim, out=out)
            out *= factor
    else:
        raise ValueError('Unknown reduction type "{}"'.format(reduction))


class _SecondOrderContractions(torch.autograd.Function):
    """Custom autograd for computation of second-order contractions.

    In order to limit the amount of memory manipulation for second-order contractions,
    we use in-place methods which are not supported by the torch autograd system.
    We must thus provide a custom autograd implementation.
    """
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, reduction='mean'):
        ctx.reduction = reduction

        dims = range(1, 4)
        dim_pairs = [(1, 2), (1, 3), (2, 3)]

        num_channels = x.shape[-1]

        stacked = x.new_empty([x.shape[0], x.shape[1], x.shape[2], num_channels * 6])

        for i, dim in enumerate(dims):
            reduction_out = stacked.narrow(-1, i * num_channels, num_channels)
            _generic_reduction(x, dim=dim, out=reduction_out, reduction=reduction)

        for i, (dima, dimb) in enumerate(dim_pairs):
            stacked.narrow(-1, (i + 3) * num_channels, num_channels)[...] = get_diag(x, dim1=dima, dim2=dimb)

        return stacked

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        n_batch = grad_output.shape[0]
        n_dim = grad_output.shape[1]
        n_channels = grad_output.shape[-1] // 6

        reduction = ctx.reduction

        grad_input = grad_output.new_zeros([n_batch, n_dim, n_dim, n_dim, n_channels])

        for i in range(3):
            if reduction == 'mean':
                factor = 1 / n_dim
            elif reduction == 'sum':
                factor = 1
            elif reduction == 'sqrt':
                factor = 1 / math.sqrt(n_dim)
            else:
                raise ValueError('unknown reduction type "{}" in backward pass.'.format(reduction))

            grad_input.add_(
                grad_output.narrow(-1, i * n_channels, n_channels).unsqueeze(i + 1),
                alpha=factor)

        dim_pairs = [(1, 2), (1, 3), (2, 3)]
        for i, (dima, dimb) in enumerate(dim_pairs):
            get_diag(grad_input, dima, dimb).add_(
                grad_output.narrow(-1, (i + 3) * n_channels, n_channels))

        return grad_input, None


def _compute_second_order_contractions(x, dims, reduction='mean'):
    """Compute second order contractions without concatenation.

    This function computes all second-order contractions of a 3rd-order tensor,
    but unlike `_compute_second_order_contraction_reference`, pre-allocates
    the output tensor to avoid allocations when concatenating all contractions.

    Note: currently this only speeds up the entire model by about 2%, however,
    there are potentially interesting gains by fusing all the contractions together.

    Parameters
    ----------
    x : torch.Tensor
        A 5-dimensional torch tensor, with first dimension batch and last dimension channels
    dims : List[int]
        A list of integers indicating which dimensions are equivariant, currently must be `[1, 2, 3]`.
    reduction : str
        One of 'mean', 'sum' or 'sqrt'. Indicates how the reduction is normalized after computation.

    Returns
    -------
    List[torch.Tensor]
        A list of the sum contractions
    List[torch.Tensor]
        A list of the diagonal elements
    torch.Tensor
        A tensor containing the concatenate elements of the previous lists.
    """
    if dims[0] != 1 or dims[1] != 2 or dims[2] != 3:
        raise ValueError('Contraction only supported for NHWDC format')

    n_channels = x.shape[-1]
    stacked = _SecondOrderContractions.apply(x, reduction)
    second_order_diags = [
        stacked.narrow(-1, i*n_channels, n_channels) for i in range(3, 6)]

    return second_order_diags, stacked


def _third_order_permutation_matmul_reference(layer, x):
    idx_perms = build_idx_3_permutations(x.shape, 1)

    with torch.autograd.profiler.record_function('assemble_permutations_3'):
        all_third_orders = torch.cat([x.permute(pi) for pi in idx_perms], dim=-1)
    third_order_mix = layer(all_third_orders)
    return third_order_mix


def _third_order_permutation_matmul(layer, x):
    """Third-order group-averaged matrix-multiplication.

    This function implements the permutation-averaged matrix-multiplication
    without copying the input matrix.

    Note: this is currently slower as we cannot avoid the copies.
    """
    idx_perms = build_idx_3_permutations(x.shape, 1)

    W = layer.weight.t()

    in_channels = x.shape[-1]
    out_channels = W.shape[-1]

    output = x.permute(idx_perms[0]).matmul(W.narrow(0, 0, in_channels))

    output_shape = output.shape

    output = output.contiguous().view(-1, out_channels)

    for i, pi in enumerate(idx_perms[1:]):
        permuted_contiguous = x.permute(pi).contiguous().view(-1, in_channels)
        output.addmm_(permuted_contiguous,  W.narrow(0, (i + 1) * in_channels, in_channels))

    output += layer.bias

    return output.view(*output_shape)


def _expand_mix_reference(mixed_full, mixed_slices, mixed_diag, pdim1, pdim2, pdim3):
    unsqueeze_dims = [(pdim1, pdim1 + 1), (pdim1, pdim3), (pdim2, pdim3)]
    for i, usd in enumerate(unsqueeze_dims):
        ms_i = mixed_slices[i]
        expanded_slice = move_to_end(ms_i, pdim2)
        expanded_slice = torch.diag_embed(expanded_slice, dim1=usd[0], dim2=usd[1])
        mixed_full += expanded_slice

    mixed_diag = move_to_end(mixed_diag, dim=pdim1)
    expanded_diag = torch.diag_embed(mixed_diag, dim1=-1, dim2=pdim1)
    expanded_diag = torch.diag_embed(expanded_diag, dim1=pdim2, dim2=pdim3)
    mixed_full += expanded_diag


def _expand_mix(mixed_full, mixed_slices, mixed_diag, pdim1, pdim2, pdim3, diagonal_free=False):
    """Implementation of expansion which directly adds into the source mixed tensor.

    This currently provides a 5% speed-up for the whole network over the reference
    implementation `_expand_mix_reference`.

    Parameters
    ----------
    mixed_full : torch.Tensor
        Full tensor into which the slices are accumulated.
    mixed_slices : List[torch.Tensor]
        List of tensors corresponding to the slices to be accumulated into the full tensor.
    mixed_diag : torch.Tensor
        Diagonal to be accumulated into the full tensor.

    diagonal_free : bool, optional
        If `True`, indicates that diagonal values should be overwritten by lower slices instead
        of accumalted.
    """
    unsqueeze_dims = [(pdim1, pdim1 + 1), (pdim1, pdim3), (pdim2, pdim3)]
    for i, usd in enumerate(unsqueeze_dims):
        ms_i = mixed_slices[i]
        expanded_slice = move_to_end(ms_i, pdim2)
        diag = mixed_full.diagonal(dim1=usd[0], dim2=usd[1])
        if diagonal_free:
            diag[...] = expanded_slice
        else:
            diag += expanded_slice

    mixed_diag = move_to_end(mixed_diag, dim=pdim1)
    diag_diag = mixed_full.diagonal(dim1=pdim2, dim2=pdim3).diagonal(dim1=-1, dim2=pdim1)

    if diagonal_free:
        diag_diag[...] = mixed_diag
    else:
        diag_diag += mixed_diag


def _get_third_order_contractions(x, dim1, reduction='mean'):
    """Computes all contractions required for third-order equivariant convolution.

    Parameters
    ----------
    x : torch.Tensor
        A 5-dimensional tensor in NHWDC format for which to compute the contraction.
    dim1 : int
        Currently must be set to 1, the first channel containing the "spatial" dimensions.
    reduction : str
        One of 'mean', 'sum' or 'sqrt', the reduction to use for the contractions.

    Returns
    -------
    torch.Tensor
        The second-order contractions of `x`
    torch.Tensor
        The first-order contractions of `x`
    torch.Tensor
        The zeroth-order contractions of `x`
    """
    if dim1 != 1:
        raise ValueError('dim1 must be set to 1 for compatibility reasons.')

    dim2 = dim1 + 1
    dim3 = dim1 + 2
    dims = [dim1, dim2, dim3]
    dim_pairs = [(dim1, dim2), (dim1, dim3), (dim2, dim3)]

    second_order_diags, stacked_second_order = _compute_second_order_contractions(x, dims, reduction=reduction)

    # Get First order terms
    first_order_double_sum = []
    for (dima, dimb) in dim_pairs:
        first_order_double_sum.append(_generic_reduction(x, dim=(dima, dimb), reduction=reduction))

    first_order_diag_sum = []
    for x_2 in second_order_diags:
        for dim in [dim1, dim2]:
            first_order_diag_sum.append(_generic_reduction(x_2, dim=dim, reduction=reduction))

    first_order_double_diag = [get_diag(second_order_diags[-1], dim1, dim2)]

    zeroth_order_triple_sum = [_generic_reduction(first_order_double_sum[0], dim=dim1, reduction=reduction)]
    zeroth_order_dss = [_generic_reduction(x_2, dim=(dim1, dim2), reduction=reduction) for x_2 in second_order_diags]
    zeroth_order_dds = [_generic_reduction(first_order_double_diag[0], dim=dim1, reduction=reduction)]

    all_first_order = first_order_double_sum + first_order_diag_sum + first_order_double_diag
    all_zeroth_order = zeroth_order_triple_sum + zeroth_order_dss + zeroth_order_dds

    stacked_first_order = torch.cat(all_first_order, dim=-1)
    stacked_zeroth_order = torch.cat(all_zeroth_order, dim=-1)

    return stacked_second_order, stacked_first_order, stacked_zeroth_order
