import math
import torch
import torch.nn as nn

from snnnet.utils import move_from_end, move_to_end, collapse_channels, channel_product


class SnConv2(nn.Module):
    """
    Convolutional layer for 2nd order permutation-equivariant objects

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
        If True, indicates that the parametrization should be performed in a diagonal-free fashion.
        Otherwise, we use the overparametrized form with full mixing.
    init_method : function, optional
        If not None, the initialization method to use for the weight parameters of the layer.
    inner_bias : bool, optional
        If True, indicates that inner linear layers should use a bias, otherwise, indicates
        that no bias should be applied to inner layers.
    """

    def __init__(self, in_channels, out_channels, in_node_channels=0, pdim1=1,
                 diagonal_free=False, init_method=None, inner_bias=True):
        super().__init__()
        # Fat mix layers output for all i, j
        self.fat_mix_2d = nn.Linear(2 * in_channels, out_channels, bias=inner_bias)
        self.fat_mix_1d = nn.Linear(3 * in_channels + in_node_channels, out_channels, bias=inner_bias)
        self.fat_mix_1dt = nn.Linear(3 * in_channels + in_node_channels, out_channels, bias=inner_bias)
        self.fat_mix_0d = nn.Linear(2 * in_channels + in_node_channels, out_channels, bias=inner_bias)

        # Diag mix layers output only on i=j
        self.diag_mix_1d = nn.Linear(3 * in_channels + in_node_channels, out_channels, bias=inner_bias)
        self.diag_mix_0d = nn.Linear(2 * in_channels + in_node_channels, out_channels, bias=inner_bias)

        if init_method is not None:
            for linear in [
                self.fat_mix_2d, self.fat_mix_1d, self.fat_mix_1dt, self.fat_mix_0d,
                self.diag_mix_1d, self.diag_mix_0d]:
                init_method(linear.weight)

        self.pdim1 = pdim1

        self.diagonal_free = diagonal_free
        self.inner_bias = inner_bias

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

        # Get all sums
        contractions = _get_contractions(
            x, dim1=pdim1, dim2=pdim2, mask=mask,
            reduction='sqrt' if self.diagonal_free else 'mean')
        rowsum, colsum, diag, rowcolsum, diag_sum = contractions

        # Build the objects to pass through the linear layers
        raw_objs_2d = torch.cat([x, x.transpose(pdim1, pdim2)], dim=-1)
        raw_objs_1d = torch.cat([rowsum, colsum, diag], dim=-1)
        raw_objs_0d = torch.cat([rowcolsum, diag_sum], dim=-1)

        if x_node is not None:
            node_sum = _get_node_contractions(x_node, pdim1, mask)
            raw_objs_1d = torch.cat([raw_objs_1d, x_node], dim=-1)
            raw_objs_0d = torch.cat([raw_objs_0d, node_sum], dim=-1)

        # Perform linear mixing along channel index and unsqueeze to right shapes
        mixed_2d = self.fat_mix_2d(raw_objs_2d)
        mixed_1d = self.fat_mix_1d(raw_objs_1d)
        mixed_1dt = self.fat_mix_1dt(raw_objs_1d)
        mixed_0d = self.fat_mix_0d(raw_objs_0d)
        diag_mixed_1d = self.diag_mix_1d(raw_objs_1d)
        diag_mixed_0d = self.diag_mix_0d(raw_objs_0d)

        # Recombine into a big tensor
        mixed_1d = mixed_1d.unsqueeze(pdim2)
        mixed_1dt = mixed_1dt.unsqueeze(pdim1)
        mixed_0d = mixed_0d.unsqueeze(pdim1).unsqueeze(pdim2)
        fat_mixed = mixed_1d + mixed_1dt + mixed_0d + mixed_2d
        diag_mixed = move_to_end(diag_mixed_1d, dim=pdim1) + diag_mixed_0d.unsqueeze(-1)

        if self.diagonal_free:
            fat_mixed *= (1 / 2)
            diag_mixed *= (1 / math.sqrt(2))

        fat_diag = fat_mixed.diagonal(dim1=pdim1, dim2=pdim2)

        if self.diagonal_free:
            fat_diag[...] = diag_mixed
        else:
            fat_diag.add_(diag_mixed)

        x_out = fat_mixed

        if mask is not None:
            x_out[~mask] = 0
        return x_out


class SnConv2to1(nn.Module):
    """
    Convolutional layer that sends a second order object to a first order object.

    Parameters
    ----------
    in_channels : int
        The size of the channel index of the incoming tensor
    out_channels : int
        The size of the channel index of the outgoing tensor
    in_node_channels : int
        Number of input node channels.
    pdim1 : int
        First dimension on which the permutation group acts.
        Second dimension on which the group acts must be pdim1+1
    diagonal_free : bool, optional
        If True, indicates that diagonal-free normalization should be used.
    init_method : function, optional
        Optional initialization function for the weights of the linear layers.
    inner_bias : bool, optional
        Whether inner linear layers should have a bias term.
    """
    def __init__(self, in_channels, out_channels, in_node_channels=0, pdim1=1, diagonal_free=False, init_method=None, inner_bias=True):
        super().__init__()
        # Fat mix layers output for all i, j
        self.fat_mix_1d = nn.Linear(3 * in_channels + in_node_channels, out_channels, bias=inner_bias)
        self.fat_mix_0d = nn.Linear(2 * in_channels + in_node_channels, out_channels, bias=inner_bias)
        self.pdim1 = pdim1

        if init_method is not None:
            init_method(self.fat_mix_0d.weight)
            init_method(self.fat_mix_1d.weight)

        self.diagonal_free = diagonal_free
        self.inner_bias = inner_bias

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
        mask : :class:`torch.Tensor`
            Mask used to account for graphs of unequal size.
            Index of the channel dimension, defaults to -1
        """
        pdim1 = self.pdim1
        pdim2 = pdim1 + 1

        # Get all sums
        contractions = _get_contractions(
            x, dim1=pdim1, dim2=pdim2, mask=mask,
            reduction='sqrt' if self.diagonal_free else 'mean')
        rowsum, colsum, diag, rowcolsum, diag_sum = contractions

        # Build the objects to pass through the linear layers
        raw_objs_1d = torch.cat([rowsum, colsum, diag], dim=-1)
        raw_objs_0d = torch.cat([rowcolsum, diag_sum], dim=-1)

        if x_node is not None:
            node_sum = _get_node_contractions(x_node, pdim1, mask)
            raw_objs_1d = torch.cat([raw_objs_1d, x_node], dim=-1)
            raw_objs_0d = torch.cat([raw_objs_0d, node_sum], dim=-1)

        # Perform linear mixing along channel index and unsqueeze to right shapes
        mixed_1d = self.fat_mix_1d(raw_objs_1d)
        mixed_0d = self.fat_mix_0d(raw_objs_0d)

        x_out = mixed_1d + mixed_0d.unsqueeze(pdim1)

        if self.diagonal_free:
            x_out *= 1 / math.sqrt(2)

        if mask is not None:
            mask_1d = torch.sum(mask, dim=pdim1).bool()
            x_out[~mask_1d] = 0
        return x_out


class Neighborhood(nn.Module):
    """
    Layer that learns a local neighborhood.

    Parameters
    ----------
    in_channels : int
        The size of the channel index of the incoming tensor
    neighb_channels : int
        Number of channels in the learned neighborhood.
    in_node_channels : int, optional
        Number of input node channels.  Default is zero.
    pdim1 : int
        First dimension on which the permutation group acts.
        Second dimension on which the group acts must be pdim1+1
    attn_nonlinearity : None or pytorch module, optional:
        Nonlinearity to use for the neighborhood.  Default is sigmoid.
    """
    def __init__(self, in_channels, neighb_channels, in_node_channels=0, pdim1=1, attn_nonlinearity=None):
        super().__init__()
        self.contraction = SnConv2to1(in_channels, neighb_channels,
                                      in_node_channels, pdim1)

        if attn_nonlinearity is None:
            self.attn_nonlinearity = nn.Sigmoid()
        else:
            self.attn_nonlinearity = attn_nonlinearity()
        self.neighb_channels = neighb_channels

    def forward(self, x, x_node=None, mask=None):
        """
        Runs a 2d convolution on the input data.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Input to the convolution, of shape B x n x n x m x C_in
            where B is the batch size, C_in is the number of channels,
            m is the number of neighborhoods, and n is the size of the permutation.
        x_node : :class:`torch.Tensor` or None, optional
            Node features to input, of shape B x n x m x C_in.
            Note that here C_in is the number of input node channels,
            which does not need to be the same as input channels for x.
        mask : :class:`torch.ByteTensor`
            Mask used to account for graphs of unequal size.
            True if a valid atom, False otherwise
        """
        node_features = self.contraction(x, x_node, mask)
        if mask is not None:
            lower_dim_mask = torch.sum(mask, dim=self.contraction.pdim1+1).bool()

        neighbs = self.attn_nonlinearity(node_features)
        # mask_sum = torch.sum(mask.float(), dim=dim2).unsqueeze(-1)
        if mask is not None:
            lower_dim_mask = torch.sum(mask, dim=self.contraction.pdim1+1).bool()
            neighbs[~lower_dim_mask] = 0
        return neighbs


class LocalConv2(nn.Module):
    """
    Layer that perfoms convolutions with a learned notion of locality.

    Parameters
    ----------
    in_channels : int
        The size of the channel index of the incoming tensor
    neighb_channels : int
        Number of channels in the learned neighborhood.
    out_channels : int
        The size of the channel index of the outgoing tensor
    in_node_channels : int, optional
        Number of input node channels.  Default is zero.
    pdim1 : int
        First dimension on which the permutation group acts.
        Second dimension on which the group acts must be pdim1+1
    attn_nonlinearity : None or pytorch module, optional:
        Nonlinearity to use for the neighborhood.  Default is sigmoid.
    """
    def __init__(self, in_channels, neighb_channels, out_channels, in_node_channels=0, pdim1=1, attn_nonlinearity=None, product_type='matmul'):
        super().__init__()
        self.neighborhood = Neighborhood(in_channels, neighb_channels, in_node_channels, pdim1, attn_nonlinearity)
        self.conv = SnConv2(in_channels * neighb_channels, out_channels, in_node_channels=in_node_channels)
        self.pdim1 = pdim1
        self.product_type = product_type


    def forward(self, x, x_node=None, mask=None):
        """
        Runs a 2d convolution on the input data.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Input to the convolution, of shape B x n x n x m x C_in
            where B is the batch size, C_in is the number of channels,
            m is the number of neighborhoods, and n is the size of the permutation.
        x_node : :class:`torch.Tensor` or None, optional
            Node features to input, of shape B x n x m x C_in.
            Note that here C_in is the number of input node channels,
            which does not need to be the same as input channels for x.
        mask : :class:`torch.ByteTensor`
            Mask used to account for graphs of unequal size.
            True if a valid atom, False otherwise
        """
        neighbs = self.neighborhood(x, x_node, mask)
        # x = torch.einsum('bijlc,bkld->bijkdc', x, neighbs)
        # x = collapse_channels(x)
        if self.product_type == 'matmul':
            x, x_node = _neighbor_matmul_product(x, x_node, neighbs, self.pdim1, mask)
        elif self.product_type == 'elementwise':
            x, x_node = _neighbor_elementwise_product(x, x_node, neighbs, self.pdim1, mask)
        else:
            raise ValueError('Product type %s not recognized!' % self.product_type)
        x_out = self.conv(x, x_node, mask=mask)
        if mask is not None:
            x_out[~mask] = 0
        return x_out


def _neighbor_matmul_product(x, x_node, neighbs, dim1, mask=None):
    if mask is not None:
        mask_sum = torch.sum(mask.float(), dim=dim1+1)
        mask_1d = mask_sum.bool()
        mask_sum = mask_sum.unsqueeze(-1)
        norm_neighbs = neighbs / mask_sum
        norm_neighbs[~mask_1d] = 0
    else:
        mask_sum = x.shape[dim1]
        norm_neighbs = neighbs / mask_sum

    x = torch.einsum('bijlc,bkld->bijkdc', x, norm_neighbs)
    x = collapse_channels(x)
    if mask is not None:
        x[~mask] = 0

    if x_node is not None:
        x_node = torch.einsum('bilc,bkld->bikdc', x_node, neighbs)
        x_node = collapse_channels(x_node)
        x_node[~mask_1d] = 0
    return x, x_node


def _neighbor_elementwise_product(x, x_node, neighbs, dim1, mask=None):
    if mask is not None:
        mask_1d = torch.sum(mask.float(), dim=dim1+1).bool()

    expanded_neighbs = neighbs.unsqueeze(dim1) * neighbs.unsqueeze(dim1+1)
    x = channel_product(x, expanded_neighbs)
    if mask is not None:
        x[~mask] = 0

    if x_node is not None:
        x_node = channel_product(x_node, neighbs**2)
        if mask is not None:
            x_node[~mask_1d] = 0
    return x, x_node


def _get_contractions(x, dim1, dim2, mask=None, reduction='mean'):
    """ Get the set of possible contractions of a second order tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor for which to compute contractions
    reduction : str
        One of 'mean', 'sum', 'sqrt', indicating how the reduction should be computed.
    """
    rowsum = torch.sum(x, dim=dim1)
    colsum = torch.sum(x, dim=dim2)
    diag = torch.diagonal(x, dim1=dim1, dim2=dim2)
    rowcolsum = torch.sum(colsum, dim=dim1)
    diag_sum = torch.sum(diag, dim=-1)
    diag = move_from_end(diag, dim1)

    # Divide by graph size.
    if mask is not None:
        mask_sum = torch.sum(mask.float(), dim=dim2).unsqueeze(-1)
        mask_double_sum = torch.sum(mask_sum, dim=dim1)
    else:
        mask_sum = x.shape[dim1]
        mask_double_sum = mask_sum**2

    if reduction == 'sum':
        mask_sum = 1
        mask_double_sum = 1
    elif reduction == 'sqrt':
        mask_sum = torch.sqrt(mask_sum) if mask is not None else math.sqrt(mask_sum)
        mask_double_sum = torch.sqrt(mask_double_sum) if mask is not None else math.sqrt(mask_double_sum)
    elif reduction == 'mean':
        pass
    else:
        raise ValueError('reduction must be one of "mean", "sum" or "sqrt".')

    rowsum /= mask_sum
    colsum /= mask_sum

    # TODO: clarify denominator here. sqrt(mask_double_sum) vs mask_double_sum
    diag_sum /= torch.sqrt(mask_double_sum) if mask is not None else math.sqrt(mask_double_sum)

    rowcolsum /= mask_double_sum

    return rowsum, colsum, diag, rowcolsum, diag_sum


def _get_node_contractions(x_node, dim1, dim2, mask=None):
    """
    Get the contraction of the node features.
    """
    if mask is not None:
        mask_sum = torch.sum(mask, dim=dim2).bool()
        mask_sum = torch.sqrt(torch.sum(mask_sum.float(), dim=dim1).unsqueeze(-1))
    else:
        mask_sum = math.sqrt(x_node.shape[dim1])

    x_node_sum = torch.sum(x_node, dim=dim1)
    x_node_sum /= mask_sum

    return x_node_sum
