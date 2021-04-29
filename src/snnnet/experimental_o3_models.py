import torch
import torch.nn as nn
from snnnet.experimental_layers import ResidualBlockO3
from snnnet.third_order_layers import expand_2_to_3
from snnnet.third_order_vae import BasicO3Layer, _build_expansion_fxn
from snnnet.third_order_vae import ContractionLayerTo1, ContractionLayerTo2
from snnnet import utils


class SnResidualEncoder(torch.nn.Module):
    """Encoder for permutation-equivariant objects based on third-order equivariant convolutions."""

    def __init__(self, in_channels: int, hidden_size: int, num_residual_blocks: int, out_channels: int, node_channels: int = 0, batch_norm: bool = True, nonlinearity=None, init_method=None):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        if nonlinearity is None:
            nonlinearity = torch.nn.ReLU()
        self.layers.append(BasicO3Layer(
            3 * in_channels, hidden_size, in_node_channels=node_channels, nonlinearity=nonlinearity, batch_norm=batch_norm, init_method=init_method))

        # for num_output_channels in hidden_channels:
        for i in range(num_residual_blocks):
            self.layers.append(ResidualBlockO3(hidden_size, batch_norm=batch_norm, init_method=init_method, nonlinearity=nonlinearity))

        self.contraction_conv = ContractionLayerTo1(hidden_size, out_channels, batch_norm, init_method=init_method)

    @utils.record_function('sn_encoder')
    def forward(self, x, x_node=None):
        x = expand_2_to_3(x, 'individual')
        for i, layer in enumerate(self.layers):
            with torch.autograd.profiler.record_function('layer{}'.format(i + 1)):
                x = layer(x, x_node)
                x_node = None  # ensure x_node only passed to first layer
        return self.contraction_conv(x)


class SnResidualDecoder(torch.nn.Module):
    """Decoder for permutation-equivariant objects, based on third-order equivariant convolutions."""

    def __init__(self, in_channels: int, hidden_size: int, num_residual_blocks: int, out_channels: int, node_out_channels=0, batch_norm=True, expand_type='direct', nonlinearity=None, init_method=None):
        super().__init__()
        self.expansion_fxn, in_channels_post_expansion = _build_expansion_fxn(in_channels, expand_type)

        self.layers = torch.nn.ModuleList()
        self.layers.append(BasicO3Layer(in_channels_post_expansion, hidden_size, nonlinearity=nonlinearity, batch_norm=batch_norm, init_method=init_method))

        for i in range(num_residual_blocks):
            self.layers.append(ResidualBlockO3(hidden_size, batch_norm=batch_norm, init_method=init_method, nonlinearity=nonlinearity))

        # self.final_conv = SnConv2(hidden_size, out_channels)
        self.final_conv = ContractionLayerTo2(hidden_size, out_channels, batch_norm=batch_norm, init_method=init_method)

        if node_out_channels > 0:
            self.final_node_conv = ContractionLayerTo1(hidden_size, node_out_channels, batch_norm, init_method=init_method)
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
            x = self.expansion_fxn(x)

        for i, layer in enumerate(self.layers):
            with torch.autograd.profiler.record_function('layer{}'.format(i + 1)):
                x = layer(x)

        x_out = self.final_conv(x)

        if self.final_node_conv is not None:
            x_node_out = self.final_node_conv(x)
        else:
            x_node_out = None

        return x_out, x_node_out
