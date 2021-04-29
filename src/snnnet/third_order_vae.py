"""Main modules for implementing a permutation-equivariant VAE through third-order expansion.
"""

import functools
import torch
from . import third_order_layers
from . import utils
from . import basic_layers


class BasicO3Layer(torch.nn.Module):
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

        self.conv = third_order_layers.SnConv3(
            in_channels, out_channels,
            in_node_channels=in_node_channels,
            diagonal_free=True,
            init_method=init_method,
            inner_bias=False)

        self.nonlinearity = nonlinearity

        if batch_norm:
            self.bn = torch.nn.BatchNorm3d(out_channels)
        else:
            self.bn = basic_layers.BiasLayer(out_channels)

    def forward(self, x, x_node=None):
        x = self.conv(x, x_node)
        # Note: batch-normalization expects NHWDC format.
        x = self.bn(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

        return self.nonlinearity(x)


class ContractionLayerTo1(torch.nn.Module):
    """Equivariant contraction layer to first order.

    This layer performs a single convolution from 3rd order to 1st order, followed
    by batch normalization. This layer does not apply any non-linearity.
    """

    def __init__(self, in_channels, out_channels, batch_norm=True, init_method=None):
        super().__init__()

        self.conv = third_order_layers.SnConv3to1(
            in_channels, out_channels, diagonal_free=True,
            init_method=init_method, inner_bias=False)

        if batch_norm:
            self.bn = torch.nn.BatchNorm1d(out_channels)
        else:
            self.bn = basic_layers.BiasLayer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)


class ContractionLayerTo2(torch.nn.Module):
    """Equivariant contraction layer to second order.

    This layer performs a single convolution from 3rd order to 2nd order, followed
    by batch normalization. This layer does not apply any non-linearity.
    """

    def __init__(self, in_channels, out_channels, batch_norm=True, init_method=None):
        super().__init__()

        self.conv = third_order_layers.SnConv3to2(
            in_channels, out_channels, diagonal_free=True,
            init_method=init_method, inner_bias=False)

        if batch_norm:
            self.bn = torch.nn.BatchNorm2d(out_channels)
        else:
            self.bn = basic_layers.BiasLayer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class SnEncoder(torch.nn.Module):
    """Encoder for permutation-equivariant objects based on third-order equivariant convolutions."""

    def __init__(self, in_channels, hidden_channels, out_channels, node_channels=0, batch_norm=True, init_method=None):
        super().__init__()

        self.layers = torch.nn.ModuleList()

        in_channels_post_expansion = 3 * in_channels
        num_current_input_channels = in_channels_post_expansion

        for num_output_channels in hidden_channels:
            self.layers.append(
                BasicO3Layer(num_current_input_channels, num_output_channels, node_channels,
                             batch_norm=batch_norm, init_method=init_method))
            node_channels = 0  # only set node channels for first layer
            num_current_input_channels = num_output_channels

        self.contraction_conv = ContractionLayerTo1(
            num_current_input_channels, out_channels, batch_norm, init_method)

    @utils.record_function('sn_encoder')
    def forward(self, x, x_node=None):
        x = third_order_layers.expand_2_to_3(x, 'individual')

        for i, layer in enumerate(self.layers):
            with torch.autograd.profiler.record_function('layer{}'.format(i + 1)):
                x = layer(x, x_node)
                x_node = None  # ensure x_node only passed to first layer

        return self.contraction_conv(x)


class SnDecoder(torch.nn.Module):
    """Decoder for permutation-equivariant objects, based on third-order equivariant convolutions."""

    def __init__(self, in_channels, hidden_channels, out_channels, node_out_channels=0, expand_type='individual', batch_norm=True, init_method=None):
        super().__init__()

        self.expansion_fxn, in_channels_post_expansion = _build_expansion_fxn(in_channels, expand_type)

        self.layers = torch.nn.ModuleList()

        num_current_in_channels = in_channels_post_expansion
        for num_out_channels in hidden_channels:
            self.layers.append(
                BasicO3Layer(num_current_in_channels, num_out_channels, batch_norm=batch_norm, init_method=init_method))
            num_current_in_channels = num_out_channels

        self.final_conv = ContractionLayerTo2(num_current_in_channels, out_channels, batch_norm, init_method)

        if node_out_channels > 0:
            self.final_node_conv = ContractionLayerTo1(num_current_in_channels, node_out_channels, batch_norm, init_method)
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


class SnVAE(torch.nn.Module):
    """VAE for permutation-equivariant objects.

    This module provides a simple wrapper to treat an encoder-decoder pair as a VAE
    by sampling a random normal latent.

    """

    def __init__(self, encoder, decoder, latent_param_transform=None):
        """Initializes a new VAE with the specified encoder and decoder.

        Parameters
        ----------
        encoder : torch.nn.Module
            Encoder to be used for the VAE. The output of the encoder is assumed to have
            an even number of channels, with the first half representing the log-variance,
            and the second half representing the mean of each latent dimension.
        decoder : torch.nn.Module
            Decoder to be used for the VAE.
        latent_param_transform : torch.nn.Module, optional
            If not None, a module to be used to transform the latent parameters prior
            to them being used to sample.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_param_transform = latent_param_transform

    def forward(self, edge_embeddings, node_embeddings, positional_embeddings=None, generator=None):
        """Obtains latent parameters from the encoder, samples, and generates predicted from decoder.

        Parameters
        ----------
        edge_embeddings
            Representation for edge data
        node_embeddings
            Representation for node data
        positional_embeddings
            Embeddings representing positional node data. This representation is
            appended to both the `node_embeddings` and the sampled latent variables.
        generator : torch.Generator, optional
            An optional `torch.Generator` to be used for sampling the latent variables.

        Returns
        -------
        prediction : Any
            The output of the decoder on the sampled latent
        mu : torch.Tensor
            Inferred posterior mean
        logvar : torch.Tensor
            Inferred posterior log-variance
        """
        if positional_embeddings is not None:
            node_embeddings = torch.cat((node_embeddings, positional_embeddings), dim=-1)

        latent_parameters = self.encoder(edge_embeddings, node_embeddings)

        if self.latent_param_transform is not None:
            latent_parameters = self.latent_param_transform(latent_parameters)

        logvar, mu = torch.split(latent_parameters, latent_parameters.shape[-1] // 2, dim=-1)
        std = logvar.mul(0.5).exp_()

        epsilon = torch.randn(std.shape, dtype=std.dtype, device=std.device, generator=generator)
        latent_sample = torch.addcmul(mu, epsilon, std)

        if positional_embeddings is not None:
            latent_sample = torch.cat((latent_sample, positional_embeddings), dim=-1)

        prediction = self.decoder(latent_sample)

        return prediction, mu, logvar


def _expand_individual(x, pdim1):
    x = third_order_layers.expand_1_to_2(x, 'individual', pdim1=pdim1)
    x = third_order_layers.expand_2_to_3(x, 'individual', pdim1=pdim1)
    return x


def _build_expansion_fxn(in_channels, expand_type='individual', pdim1=1):
    if expand_type == 'direct':
        in_channels_post_expansion = 3 * in_channels
        expansion_fxn = functools.partial(
            third_order_layers.expand_1_to_3, mode='direct', pdim1=pdim1)

    elif expand_type == 'individual':
        in_channels_post_expansion = 6 * in_channels
        expansion_fxn = functools.partial(
            _expand_individual, pdim1=pdim1)

    elif expand_type == "linear":
        in_channels_post_expansion = in_channels
        expansion_fxn = functools.partial(
            third_order_layers.expand_1_to_3, mode='linear', pdim1=pdim1)

    else:
        return ValueError("Expansion mode not recognized")
    return expansion_fxn, in_channels_post_expansion
