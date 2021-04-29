import torch
import torch.nn.functional as F


def vae_loss_function(recon_x, x, mu, logvar, pos_weight, norm=1, recon_N=None, node_labels=None, node_multiplier=1, kld_multiplier=1):
    """
    Standard VAE loss function.

    Parameters
    ----------
    recon_x : torch tensor
        Reconstruction.  Assumed to be in (0, 1)
    x : torch tensor
        Original Input.  Assumed to be (0 or 1)
    logvar : torch tensor
        log variance of the latent gaussians
    pos_weight : torch tensor
        Positive weight to be used in BCE reconstruction loss
    norm : float
        Weight in front of the BCE loss
    recon_N : torch tensor, Optional
        Node features.  Assumed to be categorical in (0, 1).
    node_labels : torch tensors, Optional
        True node features.  Assumed to be (0, 1)
    node_multiplier : float, Optional
        Multiplier for the node error. Default 1
    kld_multiplier : float, Optional
        Multiplier for the KL divergence. Default 1

    Returns
    -------
    Loss : torch tensor
        Total Loss
    node_loss : torch tensor
        Loss on the nodes.  Only returned if recon_N, node_labels given.
    BCE : torch tensor
        Loss on the graph.
    KLD: torch tensor
        KL Divergence loss.
    """
    BCE = norm * _compute_graph_BCE(recon_x, x, pos_weight)
    KLD = kld_multiplier * _gaussian_kld(mu, logvar)

    if recon_N is not None:
        node_loss = node_multiplier * _compute_node_CE(recon_N, node_labels)
        return BCE + KLD + node_loss, node_loss, BCE, KLD
    else:
        return BCE + KLD, BCE, KLD


def _compute_graph_BCE(recon_x, x, pos_weight):
    batch_size = x.shape[0]
    recon_x_flattened = recon_x.view(batch_size, -1)
    x_flattened = x.view(batch_size, -1)
    BCE = F.binary_cross_entropy_with_logits(recon_x_flattened, x_flattened, reduction='mean', pos_weight=pos_weight)
    return BCE


def _compute_node_CE(recon_N, node_labels):
    batch_size = recon_N.shape[0]
    num_nodes = recon_N.shape[1]
    recon_N_flattened = recon_N.view(batch_size * num_nodes, recon_N.shape[-1])
    node_labels_flattened = node_labels.view(batch_size * num_nodes)
    node_loss = F.cross_entropy(recon_N_flattened, node_labels_flattened)
    return node_loss


def _gaussian_kld(mu, logvar):
    # see Appendix B from VAE paper: # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    num_nodes = mu.shape[1]
    KLD = - (0.5 / num_nodes) * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


def _gaussian_hld(mu, logvar):
    # see Appendix B from VAE paper: # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    num_nodes = mu.shape[1]
    KLD = - (0.5 / num_nodes) * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD
