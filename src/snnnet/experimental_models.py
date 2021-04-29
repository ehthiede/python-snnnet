import torch
import torch.nn as nn
from snnnet.second_order_layers import SnConv2, SnConv2to1, LocalConv2
from snnnet.experimental_layers import MultLayer, ResidualBlockO2
from snnnet.third_order_layers import expand_1_to_2
from snnnet.second_order_vae import BasicO2Layer
from snnnet.second_order_vae import ContractionLayerTo1
from snnnet import utils


class SnResidualEncoder(torch.nn.Module):
    """Encoder for permutation-equivariant objects based on third-order equivariant convolutions."""

    def __init__(self, in_channels: int, hidden_size: int, num_residual_blocks: int, out_channels: int, node_channels: int = 0, batch_norm: bool = True, nonlinearity=None, init_method=None):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        if nonlinearity is None:
            nonlinearity = torch.nn.ReLU()
        self.layers.append(BasicO2Layer(
            in_channels, hidden_size, in_node_channels=node_channels, nonlinearity=nonlinearity, batch_norm=batch_norm, init_method=init_method))

        # for num_output_channels in hidden_channels:
        for i in range(num_residual_blocks):
            self.layers.append(ResidualBlockO2(hidden_size, batch_norm=batch_norm, nonlinearity=nonlinearity, init_method=init_method))

        self.contraction_conv = ContractionLayerTo1(hidden_size, out_channels, batch_norm=batch_norm, init_method=init_method)

    @utils.record_function('sn_encoder')
    def forward(self, x, x_node=None):
        for i, layer in enumerate(self.layers):
            with torch.autograd.profiler.record_function('layer{}'.format(i + 1)):
                x = layer(x, x_node)
                x_node = None  # ensure x_node only passed to first layer
        output = self.contraction_conv(x)
        # print(output[0])
        return output


class SnResidualDecoder(torch.nn.Module):
    """Decoder for permutation-equivariant objects, based on third-order equivariant convolutions."""

    def __init__(self, in_channels: int, hidden_size: int, num_residual_blocks: int, out_channels: int, node_out_channels=0, batch_norm=True, nonlinearity=None, init_method=None):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        in_channels_post_expansion = 2 * in_channels
        self.layers.append(BasicO2Layer(in_channels_post_expansion, hidden_size, nonlinearity=nonlinearity, batch_norm=batch_norm, init_method=init_method))

        for i in range(num_residual_blocks):
            self.layers.append(ResidualBlockO2(hidden_size, batch_norm=batch_norm, nonlinearity=nonlinearity, init_method=init_method))

        # self.final_conv = SnConv2(hidden_size, out_channels)
        self.final_conv = BasicO2Layer(hidden_size, out_channels, batch_norm=batch_norm, nonlinearity=nn.Identity(), init_method=init_method)

        if node_out_channels > 0:
            # Explicitly not doing batch norm on the final level... is that right?
            self.final_node_conv = ContractionLayerTo1(hidden_size, node_out_channels, batch_norm=batch_norm, init_method=init_method)
        else:
            self.final_node_conv = None

    @utils.record_function('sn_decoder')
    def forward(self, x):
        """Decodes the given input to the specified output, and potentially node output.

        Parameters
        ----------
        x : torch.Tensor
            1st-order tensor of shape NDC from which to decode

        Returns
        -------
        x_out : torch.Tensor
            The decoded result
        x_node_out : torch.Tensor
            If `node_out_channels > 0`, a tensor containing the decoded node outputs, else None.
        """
        with torch.autograd.profiler.record_function('expand'):
            x = expand_1_to_2(x, mode='individual')

        for i, layer in enumerate(self.layers):
            with torch.autograd.profiler.record_function('layer{}'.format(i + 1)):
                x = layer(x)

        x_out = self.final_conv(x)

        if self.final_node_conv is not None:
            x_node_out = self.final_node_conv(x)
        else:
            x_node_out = None

        return x_out, x_node_out


class SnMultEncoder(nn.Module):
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

    def __init__(self, in_channels, hidden_channel_list, out_channels, in_node_channels=0, nonlinearity=None, pdim1=1, skip_conn=False, all_channels=False):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.mult_layers = nn.ModuleList()
        in_channel_i = in_channels
        in_node_channel_i = in_node_channels
        for out_channel_i in hidden_channel_list:
            self.mult_layers.append(MultLayer(in_channel_i, in_channel_i, all_channels=all_channels))
            self.conv_layers.append(SnConv2(in_channel_i, out_channel_i, in_node_channels=in_node_channel_i, pdim1=pdim1))
            in_channel_i = out_channel_i
            in_node_channel_i = 0

        self.contractive_conv = SnConv2to1(in_channel_i, out_channels, in_node_channels=in_node_channel_i, pdim1=pdim1)
        if nonlinearity is None:
            nonlinearity = nn.ReLU
        self.nonlinearity = nonlinearity()
        self.pdim1 = pdim1

        self.skip_conn = skip_conn

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
        if self.skip_conn is True:
            x_all = []

        for conv, mult in zip(self.conv_layers, self.mult_layers):
            x = mult(x)
            x = conv(x, x_node=x_node, mask=mask)
            x_node = None
            x = self.nonlinearity(x)

            if self.skip_conn is True:
                x_all.append(x)

        if self.skip_conn is True:
            x = torch.cat(x_all, dim=3)

        x = self.contractive_conv(x, x_node=x_node)
        return x


class SnMultDecoder(nn.Module):
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

    def __init__(self, in_channels, hidden_channel_list, out_channels, out_node_channels=0, pdim1=1, outer_type='individual', nonlinearity=None, skip_conn=False, all_channels=True):
        super().__init__()
        self.outer_type = outer_type
        if self.outer_type == 'individual':
            in_channel_i = in_channels
        elif self.outer_type == 'all':
            in_channel_i = in_channels**2
        else:
            raise ValueError("Did not recognize outer type")

        self.conv_layers = nn.ModuleList()
        self.mult_layers = nn.ModuleList()
        for out_channel_i in hidden_channel_list:
            self.mult_layers.append(MultLayer(in_channel_i, in_channel_i, all_channels=all_channels))
            self.conv_layers.append(SnConv2(in_channel_i, out_channel_i, pdim1=pdim1))
            in_channel_i = out_channel_i
        self.final_conv = SnConv2(in_channel_i, out_channels, pdim1=pdim1)
        if out_node_channels > 0:
            self.final_node_conv = SnConv2to1(in_channel_i, out_node_channels, pdim1=pdim1)
        else:
            self.final_node_conv = None
        if nonlinearity is None:
            nonlinearity = nn.ReLU
        self.nonlinearity = nonlinearity()
        self.pdim1 = pdim1

        self.skip_conn = skip_conn

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
        if self.outer_type == 'individual':
            x_t = x.unsqueeze(pdim1+1)
            x = x.unsqueeze(pdim1)
            x = x * x_t
        elif self.outer_type == 'all':
            x_t = x.unsqueeze(pdim1+1).unsqueeze(-1)
            x = x.unsqueeze(pdim1).unsqueeze(-2)
            x = x * x_t
            new_shape = x.shape[:-2] + (x.shape[-1] * x.shape[-2],)
            x = x.view(new_shape)
        if mask is not None:
            x[~mask] = 0.

        if self.skip_conn is True:
            x_all = []

        for conv, mult in zip(self.conv_layers, self.mult_layers):
            x = mult(x)
            x = conv(x, mask=mask)
            x = self.nonlinearity(x)

            if self.skip_conn is True:
                x_all.append(x)

        if self.skip_conn is True:
            x = torch.cat(x_all, dim=3)

        x_out = self.final_conv(x, mask=mask)
        if self.final_node_conv is not None:
            x_node = self.final_node_conv(x, mask=mask)
            return x_out, x_node
        else:
            return x_out


class Sn2dEncoder(nn.Module):
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

    def __init__(self, in_channels, hidden_channel_list, out_channels, in_node_channels=0, nonlinearity=None, pdim1=1, skip_conn=False):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_channel_i = in_channels
        in_node_channel_i = in_node_channels
        for out_channel_i in hidden_channel_list:
            self.conv_layers.append(SnConv2(in_channel_i, out_channel_i, in_node_channels=in_node_channel_i, pdim1=pdim1))
            in_channel_i = out_channel_i
            in_node_channel_i = 0
        self.contractive_conv = SnConv2(in_channel_i, out_channels, in_node_channels=in_node_channel_i, pdim1=pdim1)
        if nonlinearity is None:
            nonlinearity = nn.ReLU
        self.nonlinearity = nonlinearity()
        self.pdim1 = pdim1

        self.skip_conn = skip_conn

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

        for module in self.conv_layers:
            x = module(x, x_node=x_node, mask=mask)
            x_node = None
            x = self.nonlinearity(x)

            if self.skip_conn:
                x_all.append(x)

        if self.skip_conn:
            x = torch.cat(x_all, dim=3)

        x = self.contractive_conv(x, x_node=x_node)
        return x


class Sn2dDecoder(nn.Module):
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

    def __init__(self, in_channels, hidden_channel_list, out_channels, out_node_channels=0, pdim1=1, nonlinearity=None, skip_conn=False):
        super().__init__()

        in_channel_i = in_channels
        self.conv_layers = nn.ModuleList()
        for out_channel_i in hidden_channel_list:
            self.conv_layers.append(SnConv2(in_channel_i, out_channel_i, pdim1=pdim1))
            in_channel_i = out_channel_i
        self.final_conv = SnConv2(in_channel_i, out_channels, pdim1=pdim1)
        if out_node_channels > 0:
            self.final_node_conv = SnConv2to1(in_channel_i, out_node_channels, pdim1=pdim1)
        else:
            self.final_node_conv = None
        if nonlinearity is None:
            nonlinearity = nn.ReLU
        self.nonlinearity = nonlinearity()
        self.pdim1 = pdim1

        self.skip_conn = skip_conn

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
        if self.skip_conn:
            x_all = []

        for module in self.conv_layers:
            x = module(x, mask=mask)
            x = self.nonlinearity(x)

            if self.skip_conn:
                x_all.append(x)

        if self.skip_conn:
            x = torch.cat(x_all, dim=3)

        x_out = self.final_conv(x, mask=mask)
        if self.final_node_conv is not None:
            x_node = self.final_node_conv(x, mask=mask)
            return x_out, x_node
        else:
            return x_out


class SnLocal2dEncoder(nn.Module):
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
    """

    def __init__(self, in_channels, hidden_channel_list, neighborhood_channels, out_channels, in_node_channels=0, pdim1=1, nonlinearity=None, product_type='elementwise'):
        super().__init__()
        self.local_conv_layers = nn.ModuleList()
        self.basic_conv_layers = nn.ModuleList()
        in_channel_i = in_channels
        in_node_channel_i = in_node_channels
        for n_channel_i, out_channel_i in zip(neighborhood_channels, hidden_channel_list):
            self.basic_conv_layers.append(SnConv2(in_channel_i, in_channel_i + in_node_channel_i, in_node_channels=in_node_channel_i, pdim1=pdim1))
            self.local_conv_layers.append(LocalConv2(in_channel_i + in_node_channel_i, n_channel_i, out_channel_i, in_node_channels=0, pdim1=pdim1, product_type=product_type))
            in_channel_i = out_channel_i
            in_node_channel_i = 0
        self.contractive_conv = SnConv2(in_channel_i, out_channels, in_node_channels=in_node_channel_i, pdim1=pdim1)
        if nonlinearity is None:
            nonlinearity = nn.ReLU
        self.nonlinearity = nonlinearity()
        self.pdim1 = pdim1

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
        for basic_module, local_module in zip(self.basic_conv_layers, self.local_conv_layers):
            x = basic_module(x, x_node=x_node, mask=mask)
            x_node = None
            x = local_module(x, x_node=x_node, mask=mask)
            x = self.nonlinearity(x)
            if mask is not None:
                x[~mask] = 0.
        x = self.contractive_conv(x, x_node=x_node)
        # x = torch.mean(x, dim=self.pdim1+1)
        return x


class SnLocal2dDecoder(nn.Module):
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
    """

    def __init__(self, in_channels, hidden_channel_list, neighborhood_channels, out_channels, out_node_channels=0, pdim1=1, outer_type='individual', nonlinearity=None, product_type='elementwise'):
        super().__init__()
        in_channel_i = in_channels

        self.local_conv_layers = nn.ModuleList()
        self.basic_conv_layers = nn.ModuleList()
        for n_channel_i, out_channel_i in zip(neighborhood_channels, hidden_channel_list):
            self.basic_conv_layers.append(SnConv2(in_channel_i, in_channel_i, pdim1=pdim1))
            self.local_conv_layers.append(LocalConv2(in_channel_i, n_channel_i, out_channel_i, in_node_channels=0, pdim1=pdim1, product_type=product_type))
            in_channel_i = out_channel_i
        self.final_conv = SnConv2(in_channel_i, out_channels, pdim1=pdim1)
        if out_node_channels > 0:
            self.final_node_conv = SnConv2to1(in_channel_i, out_node_channels, pdim1=pdim1)
        else:
            self.final_node_conv = None
        if nonlinearity is None:
            nonlinearity = nn.ReLU
        self.nonlinearity = nonlinearity()
        self.pdim1 = pdim1

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
        for basic_module, local_module in zip(self.basic_conv_layers, self.local_conv_layers):
            x = basic_module(x, mask=mask)
            x = local_module(x, mask=mask)
            x = self.nonlinearity(x)
            if mask is not None:
                x[~mask] = 0.
        x_out = self.final_conv(x, mask=mask)
        if self.final_node_conv is not None:
            x_node = self.final_node_conv(x, mask=mask)
            return x_out, x_node
        else:
            return x_out
