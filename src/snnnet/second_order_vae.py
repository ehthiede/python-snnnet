"""Main modules for implementing a permutation-equivariant VAE through third-order expansion.
"""

import torch
from . import second_order_layers, third_order_layers
from . import basic_layers, utils


class BasicO2Layer(torch.nn.Module):
    """Basic layer for third-order models.

    This layer performs a single convolution - bn - relu operation.
    """
    def __init__(self, in_channels, out_channels, in_node_channels=0, nonlinearity=None, batch_norm=True, init_method=None):
        """Initializes a new basic layer.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input
        out_channels : int
            Number of channels in the output
        in_node_channels : int, optional
            Number of channels in the input node tensor.
        nonlinearity : torch.nn.Module, optional
            Nonlinearity to apply to the output, if None, defaults to relu
        batch_norm : bool
            If True, indicates that batch normalization should be used,
            otherwise, no normalization is applied.
        """
        super().__init__()

        if nonlinearity is None:
            nonlinearity = torch.nn.ReLU()

        self.conv = second_order_layers.SnConv2(
            in_channels, out_channels,
            in_node_channels=in_node_channels,
            diagonal_free=True,
            init_method=init_method,
            inner_bias=False)

        self.nonlinearity = nonlinearity

        if batch_norm:
            self.bn = torch.nn.BatchNorm2d(out_channels)
        else:
            self.bn = basic_layers.BiasLayer(out_channels)

    def forward(self, x, x_node=None):
        x = self.conv(x, x_node)
        # Note: batch-normalization expects NHWDC format.
        x = self.bn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return self.nonlinearity(x)


class ContractionLayerTo1(torch.nn.Module):
    """Equivariant contraction layer to first order.

    This layer performs a single convolution from 3rd order to 1st order, followed
    by batch normalization. This layer does not apply any non-linearity.
    """
    def __init__(self, in_channels, out_channels, batch_norm=True, init_method=None):
        super().__init__()

        self.conv = second_order_layers.SnConv2to1(
            in_channels, out_channels, diagonal_free=True,
            init_method=init_method,
            inner_bias=False)

        if batch_norm:
            self.bn = torch.nn.BatchNorm1d(out_channels)
        else:
            self.bn = basic_layers.BiasLayer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)


class SecondOrderEncoder(torch.nn.Module):
    """Encoder for permutation-equivariant objects based on second-order equivariant convolutions."""
    def __init__(self, in_channels, hidden_channels, out_channels, node_channels=0, batch_norm=True, init_method=None):
        super().__init__()

        self.layers = torch.nn.ModuleList()

        in_channels_post_expansion = in_channels
        num_current_input_channels = in_channels_post_expansion

        for num_output_channels in hidden_channels:
            self.layers.append(
                BasicO2Layer(num_current_input_channels, num_output_channels, node_channels,
                             batch_norm=batch_norm, init_method=init_method))
            node_channels = 0  # only set node channels for first layer
            num_current_input_channels = num_output_channels

        self.contraction_conv = ContractionLayerTo1(num_current_input_channels, out_channels, batch_norm, init_method)

    @utils.record_function('sn_encoder')
    def forward(self, x, x_node=None):
        for i, layer in enumerate(self.layers):
            with torch.autograd.profiler.record_function('layer{}'.format(i + 1)):
                x = layer(x, x_node)
                x_node = None  # ensure x_node only passed to first layer

        return self.contraction_conv(x)


class SecondOrderDecoder(torch.nn.Module):
    """Decoder for permutation-equivariant objects, based on second-order equivariant convolutions."""
    def __init__(self, in_channels, hidden_channels, out_channels, node_out_channels=0, batch_norm=True, init_method=None):
        super().__init__()

        in_channels_post_expansion = 2 * in_channels

        self.layers = torch.nn.ModuleList()

        num_current_in_channels = in_channels_post_expansion
        for num_out_channels in hidden_channels:
            self.layers.append(
                BasicO2Layer(num_current_in_channels, num_out_channels,
                             batch_norm=batch_norm, init_method=init_method))
            num_current_in_channels = num_out_channels

        self.final_conv = BasicO2Layer(
            num_current_in_channels, out_channels, batch_norm=batch_norm,
            nonlinearity=torch.nn.Identity(), init_method=init_method)

        if node_out_channels > 0:
            self.final_node_conv = ContractionLayerTo1(
                num_current_in_channels, node_out_channels, batch_norm, init_method)
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
            x = third_order_layers.expand_1_to_2(x)

        for i, layer in enumerate(self.layers):
            with torch.autograd.profiler.record_function('layer{}'.format(i + 1)):
                x = layer(x)

        x_out = self.final_conv(x)

        if self.final_node_conv is not None:
            x_node_out = self.final_node_conv(x)
        else:
            x_node_out = None

        return x_out, x_node_out
