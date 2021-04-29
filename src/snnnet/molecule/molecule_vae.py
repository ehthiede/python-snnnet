"""Main training script for qm9-based VAE."""

import typing

from snnnet import second_order_vae, third_order_vae, experimental_models, experimental_o3_models

import torch


def vae_loss_function(reconstruction_edges, x, mu, logvar, pos_weight=None, edge_weight=1,
                      reconstruction_nodes=None, node_labels=None, node_weight=1, kl_weight=1):
    """Reconstruction + KL divergence losses summed over all elements and batch.

    Parameters
    ----------
    reconstruction_edges : torch.Tensor
        Tensor of logits corresponding to the reconstructed edges
    x : torch.Tensor
        Binary tensor containing true edges
    mu : torch.Tensor
        Tensor containing latent means
    logvar : torch.Tensor
        Tensor containing latent log-variance
    pos_weight : torch.Tensor, optional
        Weight of positive examples in edge reconstruction loss
    edge_weight : float, optional
        Weight of edge loss relative to node and KL loss.
    reconstruction_nodes : torch.Tensor, optional
        If not None, tensor of logits corresponding to reconstructed node features
    node_labels : torch.Tensor, optional
        If not None, tensor of integers corresponding to true node features
    node_weight : float, optional
        Weight of node loss relative to edge and KL loss.
    kld_weight : float, optional
        Weight of KL loss relative to edge and node loss.

    Returns
    -------
    total_loss : torch.Tensor
        Weighted total average loss over the batch
    node_loss : torch.Tensor
        Loss on node reconstruction problem
    edge_loss : torch.Tensor
        Loss on edge reconstruction problem
    divergence : torch.Tensor
        KL-divergence to prior on latent
    """
    edge_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        reconstruction_edges, x, reduction='mean', pos_weight=pos_weight)

    if reconstruction_nodes is not None:
        # swap axes as pytorch expects channels first whereas we use channels last
        node_loss = torch.nn.functional.cross_entropy(reconstruction_nodes.permute(0, 2, 1), node_labels)
    else:
        node_loss = 0.0

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    divergence = - (0.5) * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return edge_weight * edge_loss + kl_weight * divergence + node_weight * node_loss, node_loss, edge_loss, divergence


def _make_latent_transform(depth, num_latent_channels):
    """Create the pointwise latent transform as a dense network of given depth."""
    layers = []
    num_channels = 2 * num_latent_channels

    for _ in range(depth - 1):
        layers.append(torch.nn.Linear(num_channels, num_channels))
        layers.append(torch.nn.ReLU())

    if num_latent_channels > 0:
        layers.append(torch.nn.Linear(num_channels, num_channels))
    else:
        layers.append(torch.nn.Identity())

    return torch.nn.Sequential(*layers)


class MoleculeVAE(torch.nn.Module):
    """This class encapsulates the VAE and the embeddings used for node and edge features."""

    def __init__(self, vae_model, node_embedding, positional_embedding=None):
        """Create a new MoleculeVAE with the given VAE model and node embeddings.

        Parameters
        ----------
        vae_model : torch.nn.Module
            The module implementing the main VAE model.
        node_embedding : torch.nn.Module
            The module responsible for embedding the node features.
        positional_embedding : torch.nn.Module, optional
            The module responsible for embedding positional features, which
            are present in both the input and latent features.
        """
        super().__init__()
        self.vae_model = vae_model
        self.node_embedding = node_embedding
        self.positional_embedding = positional_embedding

    def forward(self, edge_features: torch.Tensor, node_features: torch.Tensor, positional_features: torch.Tensor = None, generator: torch.Generator = None):
        node_features = self.node_embedding(node_features)

        if self.positional_embedding is not None:
            positional_features = self.positional_embedding(positional_features)
        else:
            positional_features = None

        return self.vae_model(edge_features, node_features, positional_features, generator=generator)


class OneHotEmbedding(torch.nn.Module):
    """Helper module which implements a one-hot embedding layer.
    """

    def __init__(self, num_classes, dtype=torch.float32):
        super().__init__()
        self.num_classes = num_classes
        self.dtype = dtype

    def forward(self, x):
        return torch.nn.functional.one_hot(x, self.num_classes).to(self.dtype)


def _make_o2_encoder_decoder(encoder_channels: typing.List[int], decoder_channels: typing.List[int],
                             latent_channels: int, node_channels: int, node_feature_cardinality: int,
                             batch_norm: bool = True, expand_type: str = 'direct',
                             decoder_input_channels=None):
    if expand_type not in ('individual', 'direct'):
        raise ValueError('Order-2 VAE only supports individual / direct expansion')

    if decoder_input_channels is None:
        decoder_input_channels = latent_channels

    encoder = second_order_vae.SecondOrderEncoder(
        3, encoder_channels, 2 * latent_channels,
        node_channels=node_channels, batch_norm=batch_norm)

    decoder = second_order_vae.SecondOrderDecoder(
        decoder_input_channels, decoder_channels, 3,
        node_out_channels=node_feature_cardinality, batch_norm=batch_norm)

    return encoder, decoder


def _make_o3_encoder_decoder(encoder_channels: typing.List[int], decoder_channels: typing.List[int],
                             latent_channels: int, node_channels: int, node_feature_cardinality: int,
                             batch_norm: bool = True, expand_type: str = 'direct',
                             decoder_input_channels=None):
    if decoder_input_channels is None:
        decoder_input_channels = latent_channels

    encoder = third_order_vae.SnEncoder(
        3, encoder_channels, 2 * latent_channels,
        node_channels=node_channels, batch_norm=batch_norm)

    # The number of output node channels always matches the number of classes
    decoder = third_order_vae.SnDecoder(
        decoder_input_channels, decoder_channels, 3,
        node_out_channels=node_feature_cardinality,
        batch_norm=batch_norm, expand_type=expand_type)

    return encoder, decoder


def _make_o2_resid_encoder_decoder(encoder_channels: typing.List[int], decoder_channels: typing.List[int],
                                   latent_channels: int, node_channels: int, node_feature_cardinality: int,
                                   batch_norm: bool = True, expand_type: str = 'direct',
                                   decoder_input_channels=None):
    # The encoder / decoder channel list is slightly inappropriate for residual networks
    # These checks are a hack to make the API work.

    _check_hidden_sizes(encoder_channels, decoder_channels)

    if decoder_input_channels is None:
        decoder_input_channels = latent_channels

    num_encoder_resid_blocks = (len(encoder_channels) - 1) // 2
    num_decoder_resid_blocks = (len(decoder_channels) - 1) // 2

    encoder = experimental_models.SnResidualEncoder(
        3, decoder_channels[0], num_decoder_resid_blocks, 2*latent_channels,
        node_channels=node_channels, batch_norm=batch_norm)

    decoder = experimental_models.SnResidualDecoder(
        decoder_input_channels, decoder_channels[0], num_encoder_resid_blocks, 3,
        node_out_channels=node_feature_cardinality, batch_norm=batch_norm)

    return encoder, decoder


def _make_o3_resid_encoder_decoder(encoder_channels: typing.List[int], decoder_channels: typing.List[int],
                                   latent_channels: int, node_channels: int, node_feature_cardinality: int,
                                   batch_norm: bool = True, expand_type: str = 'direct',
                                   decoder_input_channels=None):
    # The encoder / decoder channel list is slightly inappropriate for residual networks
    # These checks are a hack to make the API work.

    _check_hidden_sizes(encoder_channels, decoder_channels)

    if decoder_input_channels is None:
        decoder_input_channels = latent_channels

    num_encoder_resid_blocks = (len(encoder_channels) - 1) // 2
    num_decoder_resid_blocks = (len(decoder_channels) - 1) // 2

    encoder = experimental_o3_models.SnResidualEncoder(
        3, decoder_channels[0], num_decoder_resid_blocks, 2*latent_channels,
        node_channels=node_channels, batch_norm=batch_norm)

    decoder = experimental_o3_models.SnResidualDecoder(
        decoder_input_channels, decoder_channels[0], num_encoder_resid_blocks, 3,
        node_out_channels=node_feature_cardinality, batch_norm=batch_norm)

    return encoder, decoder


def _check_hidden_sizes(encoder_channels, decoder_channels):
    assert(len(encoder_channels) % 2 == 1), "Num hidden channels isn't odd"
    all_ec_hidden_equal_size = all([ec_i == encoder_channels[0] for ec_i in encoder_channels])
    assert all_ec_hidden_equal_size, "all hidden channels are not the same"
    assert(len(encoder_channels) % 2 == 1), "Num hidden channels isn't odd"
    all_dc_hidden_equal_size = all([dc_i == decoder_channels[0] for dc_i in decoder_channels])
    assert all_dc_hidden_equal_size, "all hidden channels are not the same"


CORE_ARCHITECTURE = {
    'o2': _make_o2_encoder_decoder,
    'o3': _make_o3_encoder_decoder,
    'o2residual': _make_o2_resid_encoder_decoder,
    'o3residual': _make_o3_resid_encoder_decoder
}


def make_molecule_vae(encoder_channels: typing.List[int], decoder_channels: typing.List[int],
                      latent_channels: int, node_feature_cardinality: int, latent_transform_depth: int = 0,
                      node_embedding_dimension: int = 0, positional_embedding_dimension: int = 0,
                      batch_norm: bool = True, expand_type: str = 'individual',
                      architecture: str = 'o3', max_size: int = None):
    """Creates the molecule VAE model according to the given parameters.

    Parameters
    ----------
    encoder_channels
        A list of integers representing the number of channels in the encoder for each layer.
    decoder_channels
        A list of integers representing the number of channels in the decoder for each layer.
    latent_channels
        The number of channels to use in the latent space.
    node_feature_cardinality
        The cardinality of the given node feature.
    latent_transform_depth
        The number of layers in the pointwise latent transform.
    node_embedding_dimension
        The dimension of the node embedding to use. If <= 0, indicates that no embedding should
        be used, and revert to one-hot encoding.
    positional_embedding_dimension
        The dimension of the positional embedding to use. If <= 0, indicates that no positional
        embedding should be used.
    batch_norm
        Whether to use batch-normalization in the 3rd-order equivariant layers.
    expand_type
        The type of expansion to use when lifting the 1st-order latent to third-order.
        Must be one of 'individual', 'linear', or 'direct'.
    architecture
        The underlying architecture to use for the network.
        Must be one of 'o2' (second-order) or 'o3' (third-order).
    max_size
        The maximum size of the molecules considered by the VAE. Must be provided
        if `positional_embedding_dimension` is not 0.

    Returns
    -------
    MoleculeVAE
        A MoleculeVAE model created according to the specifications.
    """
    # The number of input node channels depends on whether we use node embeddings
    node_channels = node_embedding_dimension if node_embedding_dimension > 0 else node_feature_cardinality
    decoder_input_channels = latent_channels

    if positional_embedding_dimension > 0:
        node_channels += positional_embedding_dimension
        decoder_input_channels += positional_embedding_dimension

    encoder_decoder_factory = CORE_ARCHITECTURE[architecture]

    encoder, decoder = encoder_decoder_factory(
        encoder_channels, decoder_channels, latent_channels, node_channels,
        node_feature_cardinality, batch_norm, expand_type, decoder_input_channels)

    latent_transform = _make_latent_transform(latent_transform_depth, latent_channels)
    vae_model = third_order_vae.SnVAE(encoder, decoder, latent_transform)

    if node_embedding_dimension > 0:
        node_embedding = torch.nn.Embedding(node_feature_cardinality, node_embedding_dimension)
    else:
        node_embedding = OneHotEmbedding(node_feature_cardinality)

    if positional_embedding_dimension > 0:
        if max_size is None:
            raise ValueError('max_size must be provided when using positional embeddings')

        # index 0 is assumed to be a padding index for atoms that are not present
        positional_embedding = torch.nn.Embedding(max_size + 1, positional_embedding_dimension)
    else:
        positional_embedding = None

    model = MoleculeVAE(vae_model, node_embedding, positional_embedding)
    return model
