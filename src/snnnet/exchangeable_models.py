import torch.nn as nn
import torch
import torch.nn.functional as F


class SnBoolVAE(nn.Module):
    """
    Permutation Variational Autoencoder
    Built with help from https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py
    """

    def __init__(self, encoder, decoder, temperature, num_gumbel, pdim1=1):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.temperature = temperature
        self.num_gumbel = num_gumbel
        self.pdim1 = pdim1

    def forward(self, x, x_node=None, mask=None, print_hiddens=False):
        # Encoder
        encoded_vars = self.encoder(x, x_node, mask)

        log_pis = encoded_vars[..., :self.num_gumbel]
        other_vars = encoded_vars[..., self.num_gumbel:]
        N_gauss_channels = other_vars.shape[-1]
        assert(N_gauss_channels % 2 == 0), "Number of channels is not even!"
        log_var = other_vars[..., :N_gauss_channels//2]
        z_mu = other_vars[..., N_gauss_channels//2:]

        # Reparameterization trick on the gumbel bits
        x_gbl_sample = gumbel_softmax_sample(log_pis, self.temperature, softmax_dim=self.pdim1)

        # Re-parameterization trick for the gaussian bits: Sample from the distribution having latent parameters z_mu, log_var
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)
        if print_hiddens:
            print(x_gbl_sample)
        x_sample = torch.cat([x_gbl_sample, x_sample], dim=-1)

        # Decoder
        predict = self.decoder(x_sample, mask)
        return predict, z_mu, log_var, F.softmax(log_pis, dim=self.pdim1)


class SnChannelBoolVAE(nn.Module):
    """
    Permutation Variational Autoencoder
    Built with help from https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py
    """

    def __init__(self, encoder, decoder, temperature, num_gumbel_per_cat, num_cats):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.temperature = temperature
        self.num_gumbel_per_cat = num_gumbel_per_cat
        self.num_cats = num_cats
        self.num_gumbel_channels = num_cats * num_gumbel_per_cat

    def forward(self, x, x_node=None, mask=None, print_hiddens=False):
        # Encoder
        encoded_vars = self.encoder(x, x_node, mask)

        log_pis = encoded_vars[..., :self.num_gumbel_channels]
        old_shape = log_pis.shape
        new_shape = old_shape[:-1] + (self.num_cats, self.num_gumbel_per_cat)
        log_pis = log_pis.view(new_shape)
        # Reparameterization trick on the gumbel bits
        x_gbl_sample = gumbel_softmax_sample(log_pis, self.temperature, softmax_dim=-1)
        x_gbl_sample = x_gbl_sample.view(old_shape)

        other_vars = encoded_vars[..., self.num_gumbel_channels:]
        N_gauss_channels = other_vars.shape[-1]
        assert(N_gauss_channels % 2 == 0), "Number of channels is not even!"
        log_var = other_vars[..., :N_gauss_channels//2]
        z_mu = other_vars[..., N_gauss_channels//2:]

        # Re-parameterization trick for the gaussian bits: Sample from the distribution having latent parameters z_mu, log_var
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)
        if print_hiddens:
            print(x_gbl_sample)
        x_sample = torch.cat([x_gbl_sample, x_sample], dim=-1)

        # Decoder
        predict = self.decoder(x_sample, mask)
        return predict, z_mu, log_var, F.softmax(log_pis, dim=-1).view(old_shape)


def bool_vae_loss(recon_x, x, qy, z_mu, log_var, norm, recon_N=None, node_labels=None, node_multiplier=1, pos_weight=1, beta_n=1, beta_g=1, pdim1=1):
    batch_size = x.shape[0]
    recon_x_flattened = recon_x.view(batch_size, -1)
    x_flattened = x.view(batch_size, -1)

    # BCE on adjacency matrix
    BCE = norm * F.binary_cross_entropy_with_logits(recon_x_flattened, x_flattened, reduction='mean', pos_weight=pos_weight)

    num_nodes = x.shape[pdim1]
    KLD_normal = - (0.5 / num_nodes) * torch.mean(1 + log_var - z_mu.pow(2) - log_var.exp())
    log_ratio = torch.log(qy * num_nodes + 1e-20)
    KLD_gumbel = torch.sum(qy * log_ratio, dim=pdim1).mean()

    loss = BCE + beta_n * KLD_normal + beta_g * KLD_gumbel

    # BCE on nodes
    if recon_N is not None:
        num_nodes = x.shape[1]
        recon_N_flattened = recon_N.view(batch_size * num_nodes, recon_N.shape[-1])
        node_labels_flattened = node_labels.view(batch_size * num_nodes)
        node_loss = node_multiplier * F.cross_entropy(recon_N_flattened, node_labels_flattened)
        loss += node_loss

    if recon_N is not None:
        return loss, node_loss, BCE, KLD_normal, KLD_gumbel
    else:
        return loss, BCE, KLD_normal, KLD_gumbel


def sample_gumbel(shape, eps=1e-20, device='cuda'):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, softmax_dim=1):
    y = logits + sample_gumbel(logits.shape, device=logits.device)
    return F.softmax(y / temperature, dim=softmax_dim)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature, device=logits.device)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard
