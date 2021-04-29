import torch
import torch.nn as nn
from snnnet.utils import move_from_end, move_to_end, collapse_channels, channel_product
from snnnet.second_order_layers import SnConv2
from snnnet.third_order_layers import SnConv3
from snnnet.basic_layers import BiasLayer


class ResidualBlockO2(nn.Module):
    """
    Residual block for a second order Sn-net.
    """
    def __init__(self, in_channels, hidden_channels=None, in_node_channels=0, nonlinearity=None, batch_norm=True, init_method=None):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = in_channels

        self.conv1 = SnConv2(
            in_channels, hidden_channels,
            in_node_channels=in_node_channels,
            diagonal_free=True,
            inner_bias=False,
            init_method=init_method)

        self.conv2 = SnConv2(
            hidden_channels, in_channels,
            in_node_channels=0,
            diagonal_free=True,
            inner_bias=False,
            init_method=init_method)

        if batch_norm:
            self.bn1 = torch.nn.BatchNorm2d(hidden_channels)
            self.bn2 = torch.nn.BatchNorm2d(in_channels)
        else:
            self.bn1 = BiasLayer(hidden_channels)
            self.bn2 = BiasLayer(in_channels)

        if nonlinearity is None:
            nonlinearity = torch.nn.ReLU()
        self.nonlinearity = nonlinearity

    def forward(self, x, x_node=None):
        y = self.conv1(x, x_node)
        y = self.bn1(y.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        y = self.nonlinearity(y)
        y = self.conv2(y)
        y = self.bn2(y.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.nonlinearity(x + y)


class ResidualBlockO3(nn.Module):
    """
    Residual block for a second order Sn-net.
    """
    def __init__(self, in_channels, hidden_channels=None, in_node_channels=0, nonlinearity=None, batch_norm=True, init_method=None):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = in_channels

        self.conv1 = SnConv3(
            in_channels, hidden_channels,
            in_node_channels=in_node_channels,
            diagonal_free=True,
            inner_bias=False,
            init_method=init_method)

        self.conv2 = SnConv3(
            hidden_channels, in_channels,
            in_node_channels=0,
            diagonal_free=True,
            inner_bias=False,
            init_method=init_method)

        if batch_norm:
            self.bn1 = torch.nn.BatchNorm3d(hidden_channels)
            self.bn2 = torch.nn.BatchNorm3d(in_channels)
        else:
            self.bn1 = BiasLayer(hidden_channels)
            self.bn2 = BiasLayer(in_channels)

        if nonlinearity is None:
            nonlinearity = torch.nn.ReLU()
        self.nonlinearity = nonlinearity

    def forward(self, x, x_node=None):
        y = self.conv1(x, x_node)
        y = self.bn1(y.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        y = self.nonlinearity(y)
        y = self.conv2(y)
        y = self.bn2(y.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        return self.nonlinearity(x + y)


# class AttentionLayerO2(nn.Module):
#     """
#     Attention block for a second order Sn-net.
#     """
#     def __init__(self, in_channels, out_channels, in_node_channels=0, attn_nonlinearity=None, batch_norm=True):


class MultLayer(nn.Module):
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
    """

    def __init__(self, in_channels, out_channels, pdim1=1, all_channels=True):
        super().__init__()
        if all_channels:
            self.mix = nn.Linear(in_channels**2 + in_channels, out_channels)
        else:
            self.mix = nn.Linear(2 * in_channels, out_channels)
        self.all_channels = all_channels
        self.pdim1 = pdim1

    def forward(self, x):
        if self.all_channels:
            x_mult = torch.einsum('bijc,bjkd->bikcd', x, x) / x.shape[1]
            x_mult = torch.flatten(x_mult, start_dim=-2)
            x = torch.cat([x_mult, x], dim=-1)
        else:
            x_mult = torch.einsum('bijc,bjkc->bikc', x, x) / x.shape[1]
            x = torch.cat([x_mult, x], dim=-1)
        return self.mix(x)


class GraphConv1D(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(GraphConv1D, self).__init__(**kwargs)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.matmul(adj, x)
        return x


class GraphConv2D(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(GraphConv2D, self).__init__(**kwargs)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.einsum('bij,bjkl->bikl', adj, x)
        return x
