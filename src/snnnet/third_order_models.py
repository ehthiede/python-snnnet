import torch
import torch.nn as nn
from snnnet.second_order_layers import SnConv2
from snnnet.third_order_layers import SnConv3, SnConv3to2, SnConv3to1, expand_2_to_3, expand_1_to_2, expand_1_to_3


class SnEncoder3(nn.Module):
    """
    Second order encoder.  Encodes into a order-one vector

    Parameters
    ----------
    in_channels : int
        The size of the channel index of the input tensor
    hidden_channel_list : iterable of ints
        Number of channels in every hidden layer of the encoder
        Length corresponds to the number of layers.
    out_channels : int
        number of output channels.
    in_node_channels : int
        Number of input node channels.
    nonlinearity : None or pytorch module, optional:
        Nonlinearity to use.  If None, defaults to ReLU
    skip_conn : Skip connection, optional.
    """

    def __init__(self, in_channels, hidden_channel_list, out_channels, in_node_channels=0, nonlinearity=None, pdim1=1, expand_mode='individual', skip_conn=False, contraction_norm=None, mixture_norm=None, lead_in=None, input_is_o2=True):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_channel_i = in_channels
        if input_is_o2:
            if lead_in is not None:
                self.lead_in_layer = SnConv2(in_channel_i, lead_in, in_node_channels=in_node_channels, pdim1=pdim1)
                in_channel_i = calc_num_expand_channels(lead_in, expand_mode=expand_mode)
                in_node_channels = 0
            else:
                self.lead_in_layer = None
                in_channel_i = calc_num_expand_channels(in_channel_i, expand_mode=expand_mode)

        if skip_conn:
            sum_out_channel_i = 0

        for out_channel_i in hidden_channel_list:
            self.conv_layers.append(SnConv3(in_channel_i, out_channel_i, pdim1=pdim1, contraction_norm=contraction_norm, mixture_norm=mixture_norm, in_node_channels=in_node_channels))
            in_node_channels = 0
            in_channel_i = out_channel_i

            if skip_conn:
                sum_out_channel_i += out_channel_i

        if skip_conn:
            # self.contractive_conv = SnConv3to1(sum_out_channel_i, out_channels, pdim1=pdim1, contraction_norm=contraction_norm, mixture_norm=mixture_norm)
            self.contractive_conv = SnConv3to1(sum_out_channel_i, out_channels, pdim1=pdim1)
        else:
            self.contractive_conv = SnConv3to1(hidden_channel_list[-1], out_channels, pdim1=pdim1)

        if nonlinearity is None:
            nonlinearity = nn.ReLU
        self.nonlinearity = nonlinearity()
        self.pdim1 = pdim1
        self.expand_mode = expand_mode

        self.skip_conn = skip_conn
        self.input_is_o2 = input_is_o2

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
        if self.skip_conn:
            x_all = []

        if self.input_is_o2:
            if self.lead_in_layer is not None:
                x = self.lead_in_layer(x, x_node=x_node)
                x_node = None

            x = expand_2_to_3(x, self.expand_mode)
        for module in self.conv_layers:
            x = module(x, x_node=x_node, mask=mask)
            x_node = None
            x = self.nonlinearity(x)

            if self.skip_conn:
                x_all.append(x)

        if self.skip_conn:
            x = torch.cat(x_all, dim=4)

        x = self.contractive_conv(x)
        return x


class SnDecoder3(nn.Module):
    """
    Second order encoder.  Encodes into a order-one vector

    Parameters
    ----------
    in_channels : int
        The size of the channel index of the input tensor
    hidden_channel_list : iterable of ints
        Number of channels in every hidden layer of the encoder
        Length corresponds to the number of layers.
    out_channels : int
        number of output channels.
    out_node_channels : int
        Number of node channels to output
    pdim1 : int
        First dimension on which the permutation group acts.
        Second dimension on which the group acts must be pdim1+1
    outer_type : string, optional
        How to handle the channels in the tensor product. Options are
        'individual' which maps to v_ic -> v_ic * v_jc, and "all" which maps
        v_ic -> v_ic * v_jd
    nonlinearity : None or pytorch module, optional:
        Nonlinearity to use.  If None, defaults to ReLU
    skip_conn : Skip connection, optional.
    """

    def __init__(self, in_channels, hidden_channel_list, out_channels, out_node_channels=0, pdim1=1, outer_type='individual', nonlinearity=None, expand_mode='individual', contraction_norm=None, mixture_norm=None, lead_in=None, output_is_o2=True):
        super().__init__()
        if expand_mode == 'direct':
            in_channel_i = 3 * in_channels
        else:
            self.outer_type = outer_type
            if self.outer_type == 'individual':
                in_channel_i = 2 * in_channels
            elif self.outer_type == 'all':
                in_channel_i = in_channels**2 + in_channels
            else:
                raise ValueError("Did not recognize outer type")

    # self.conv_layers.append(SnConv2(in_channel_i, hidden_channel_list[0], pdim1=pdim1))
            if lead_in is not None:
                self.lead_in_layer = SnConv2(in_channel_i, lead_in, pdim1=pdim1)
                in_channel_i = lead_in
            else:
                self.lead_in_layer = None

            in_channel_i = calc_num_expand_channels(in_channel_i, expand_mode=expand_mode)
        self.conv_layers = nn.ModuleList()
        for out_channel_i in hidden_channel_list:
            self.conv_layers.append(SnConv3(in_channel_i, out_channel_i, pdim1=pdim1, contraction_norm=contraction_norm, mixture_norm=mixture_norm))
            in_channel_i = out_channel_i

        self.output_is_o2 = output_is_o2
        if self.output_is_o2:
            self.final_conv = SnConv3to2(in_channel_i, out_channels, pdim1=pdim1, contraction_norm=contraction_norm, mixture_norm=mixture_norm)
        else:
            self.final_conv = SnConv3(in_channel_i, out_channels, pdim1=pdim1, contraction_norm=contraction_norm, mixture_norm=mixture_norm)

        if out_node_channels > 0:
            self.final_node_conv = SnConv3to1(in_channel_i, out_node_channels, pdim1=pdim1)
        else:
            self.final_node_conv = None
        if nonlinearity is None:
            nonlinearity = nn.ReLU
        self.nonlinearity = nonlinearity()
        self.pdim1 = pdim1
        self.expand_mode = expand_mode

    def forward(self, x, mask=None):
        """
        Runs a 2d convolution on the input data.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Input to the convolution, of shape B x n x n x C_in
            where B is the batch size, C_in is the number of channels,
            and n is the size of the permutation
        mask : :class:`torch.Tensor`
            Mask used to account for graphs of unequal size.
            Index of the channel dimension, defaults to -1
        pdim1 : int
            First dimension on which the permutation group acts.
            Second dimension on which the group acts must be pdim1+1
        """
        pdim1 = self.pdim1
        if self.expand_mode == 'direct':
            x = expand_1_to_3(x, pdim1=pdim1)
        else:
            x = expand_1_to_2(x, self.outer_type, pdim1=pdim1)
            if mask is not None:
                x[~mask] = 0.

            if self.lead_in_layer is not None:
                x = self.lead_in_layer(x)
            x = expand_2_to_3(x, self.expand_mode, pdim1=pdim1)

        for module in self.conv_layers:
            x = module(x, mask=mask)
            x = self.nonlinearity(x)

        x_out = self.final_conv(x, mask=mask)
        if self.final_node_conv is not None:
            x_node = self.final_node_conv(x, mask=mask)
            return x_out, x_node
        else:
            return x_out


def calc_num_expand_channels(orig_in_channel, expand_mode):
    if expand_mode == "individual":
        in_channel_i = 3 * orig_in_channel
    elif expand_mode == "subdivide":
        subc = (orig_in_channel // 3)
        in_channel_i = subc * subc * (orig_in_channel - 2 * subc)
        in_channel_i += subc * subc
        in_channel_i += orig_in_channel
    elif expand_mode == "all":
        subc = orig_in_channel
        in_channel_i = subc * subc * subc
        in_channel_i += subc * subc
        in_channel_i += orig_in_channel
    elif expand_mode == "linear":
        in_channel_i = orig_in_channel
    else:
        raise ValueError("Expand mode not recognized")
    return in_channel_i
