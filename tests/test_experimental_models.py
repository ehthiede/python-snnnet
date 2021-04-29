import pytest
import torch
from snnnet.exchangeable_models import SnBoolVAE, SnChannelBoolVAE, bool_vae_loss
from snnnet.second_order_models import SnEncoder, SnDecoder
from snnnet.experimental_models import SnResidualEncoder, SnResidualDecoder


@pytest.mark.parametrize('input_channels', [1, 5])
@pytest.mark.parametrize('hidden_channels', [2, 5])
@pytest.mark.parametrize('num_residual', [0, 2, 5])
@pytest.mark.parametrize('num_node_channels', [0, 3])
@pytest.mark.parametrize('batch_norm', [True, False])
def test_residual_encoder(input_channels, hidden_channels, num_residual, batch_norm, num_node_channels):
    out_channels = 2
    n = 6

    encoder = SnResidualEncoder(in_channels=input_channels, hidden_size=hidden_channels, num_residual_blocks=num_residual,
                                out_channels=out_channels, node_channels=num_node_channels, batch_norm=batch_norm)

    x = torch.randn(4, n, n, input_channels)
    pidx = torch.randperm(n)  # random permutation
    x_perm = x[:, pidx][:, :, pidx]

    if num_node_channels > 0:
        x_node = torch.randn(4, n, num_node_channels)
        x_node_perm = x_node[:, pidx]

        x_out = encoder(x, x_node=x_node)
        x_perm_out = encoder(x_perm, x_node=x_node_perm)
    else:
        x_out = encoder(x)
        x_perm_out = encoder(x_perm)

    assert(torch.norm(x_out[:, pidx] - x_perm_out) < 1e-3)


@pytest.mark.parametrize('input_channels', [1, 5])
@pytest.mark.parametrize('hidden_channels', [2, 5])
@pytest.mark.parametrize('num_residual', [0, 2, 5])
@pytest.mark.parametrize('num_node_out_channels', [0, 3])
@pytest.mark.parametrize('batch_norm', [True, False])
def test_residual_decoder(input_channels, hidden_channels, num_residual, num_node_out_channels, batch_norm):
    out_channels = 2
    n = 6

    decoder = SnResidualDecoder(in_channels=input_channels, hidden_size=hidden_channels, num_residual_blocks=num_residual,
                                out_channels=out_channels, node_out_channels=num_node_out_channels, batch_norm=batch_norm)

    x = torch.randn(4, n, input_channels)
    pidx = torch.randperm(n)  # random permutation
    x_perm = x[:, pidx]

    if num_node_out_channels > 0:
        x_out, x_node_out = decoder(x)
        x_perm_out, x_perm_node_out = decoder(x_perm)
    else:
        x_out = decoder(x)[0]
        x_perm_out = decoder(x_perm)[0]

    assert(torch.norm(x_out[:, pidx][:, :, pidx] - x_perm_out) < 1e-3)
    if num_node_out_channels > 0:
        assert(torch.norm(x_node_out[:, pidx] - x_perm_node_out) < 1e-3)


class TestBoolVAE(object):
    @pytest.mark.parametrize('input_channels', [1, 5])
    @pytest.mark.parametrize('num_gumbel', [1, 3])
    @pytest.mark.parametrize('num_normal', [1, 4])
    @pytest.mark.parametrize('temp', [0.2, 1, 2])
    def test_equivariance(self, input_channels, num_gumbel, num_normal, temp):
        n = 6
        pidx = torch.randperm(n)  # random permutation
        x = torch.randn(4, n, n, input_channels)
        x_perm = x[:, pidx][:, :, pidx]
        x_node = x_node_perm = None

        encoder = SnEncoder(input_channels, [3], num_gumbel + 2 * num_normal)
        decoder = SnDecoder(num_gumbel + num_normal, [3], input_channels)
        snc = SnBoolVAE(encoder, decoder, temp, num_gumbel)

        # Permute the output
        x_out, z_mu, log_var, qy = snc(x, x_node=x_node)
        z_mu_perm = z_mu[:, pidx]
        log_var_perm = log_var[:, pidx]
        qy_perm = qy[:, pidx]

        # Permute the input
        x_perm_out, perm_z_mu, perm_log_var, perm_qy = snc(x_perm, x_node=x_node_perm)
        assert(torch.norm(log_var_perm - perm_log_var) < 1E-3)
        assert(torch.norm(qy_perm - perm_qy) < 1E-3)
        assert(torch.norm(z_mu_perm - perm_z_mu) < 1E-3)

    @pytest.mark.parametrize('node_channels', [0, 2, 3])
    @pytest.mark.parametrize('input_channels', [1, 5])
    @pytest.mark.parametrize('num_gumbel', [1, 3])
    @pytest.mark.parametrize('num_normal', [1, 4])
    @pytest.mark.parametrize('temp', [0.2, 1, 2])
    def test_loss(self, node_channels, input_channels, num_gumbel, num_normal, temp):
        n = 6
        x = torch.randn(4, n, n, input_channels)
        if node_channels > 0:
            x_node = torch.randn(4, n, node_channels)
            node_labels = torch.randint(0, node_channels-1, (4, n))
        else:
            x_node = None
            node_labels = None

        encoder = SnEncoder(input_channels, [3], num_gumbel + 2 * num_normal, in_node_channels=node_channels)
        decoder = SnDecoder(num_gumbel + num_normal, [3], input_channels, out_node_channels=node_channels)
        snc = SnBoolVAE(encoder, decoder, temp, num_gumbel)

        # Permute the output
        if node_channels > 0:
            (x_out, x_node), z_mu, log_var, qy = snc(x, x_node=x_node)
        else:
            x_out, z_mu, log_var, qy = snc(x, x_node=x_node)
        pos_weight = torch.tensor([1])
        bool_vae_loss(x_out, x, qy, z_mu, log_var, norm=1, recon_N=x_node, node_labels=node_labels, node_multiplier=1, pos_weight=pos_weight, beta_n=1, beta_g=1)


class TestChannelBoolVAE(object):
    @pytest.mark.parametrize('input_channels', [1, 5])
    @pytest.mark.parametrize('num_gumbel_per_cat', [1, 3])
    @pytest.mark.parametrize('num_cats', [1, 2])
    @pytest.mark.parametrize('num_normal', [1, 4])
    @pytest.mark.parametrize('temp', [0.2, 1, 2])
    def test_equivariance(self, input_channels, num_gumbel_per_cat, num_cats, num_normal, temp):
        n = 6
        pidx = torch.randperm(n)  # random permutation
        x = torch.randn(4, n, n, input_channels)
        x_perm = x[:, pidx][:, :, pidx]
        x_node = x_node_perm = None

        encoder = SnEncoder(input_channels, [3], num_gumbel_per_cat * num_cats + 2 * num_normal)
        decoder = SnDecoder(num_gumbel_per_cat * num_cats + num_normal, [3], input_channels)
        snc = SnChannelBoolVAE(encoder, decoder, temp, num_gumbel_per_cat, num_cats)

        # Permute the output
        x_out, z_mu, log_var, qy = snc(x, x_node=x_node)
        z_mu_perm = z_mu[:, pidx]
        log_var_perm = log_var[:, pidx]
        qy_perm = qy[:, pidx]

        # Permute the input
        x_perm_out, perm_z_mu, perm_log_var, perm_qy = snc(x_perm, x_node=x_node_perm)
        assert(torch.norm(log_var_perm - perm_log_var) < 1E-3)
        assert(torch.norm(qy_perm - perm_qy) < 1E-3)
        assert(torch.norm(z_mu_perm - perm_z_mu) < 1E-3)

    @pytest.mark.parametrize('node_channels', [0, 2, 3])
    @pytest.mark.parametrize('input_channels', [1, 5])
    @pytest.mark.parametrize('num_gumbel_per_cat', [1, 3])
    @pytest.mark.parametrize('num_cats', [1, 2])
    @pytest.mark.parametrize('num_normal', [1, 4])
    @pytest.mark.parametrize('temp', [0.2, 1, 2])
    def test_loss(self, node_channels, input_channels, num_gumbel_per_cat, num_cats, num_normal, temp):
        n = 6
        x = torch.randn(4, n, n, input_channels)
        if node_channels > 0:
            x_node = torch.randn(4, n, node_channels)
            node_labels = torch.randint(0, node_channels-1, (4, n))
        else:
            x_node = None
            node_labels = None
        
        print(x.shape)
        encoder = SnEncoder(input_channels, [3], num_gumbel_per_cat * num_cats + 2 * num_normal, in_node_channels=node_channels)
        decoder = SnDecoder(num_gumbel_per_cat * num_cats + num_normal, [3], input_channels, out_node_channels=node_channels)
        snc = SnChannelBoolVAE(encoder, decoder, temp, num_gumbel_per_cat, num_cats)

        # Permute the output
        if node_channels > 0:
            (x_out, x_node), z_mu, log_var, qy = snc(x, x_node=x_node)
        else:
            x_out, z_mu, log_var, qy = snc(x, x_node=x_node)
        pos_weight = torch.tensor([1])
        bool_vae_loss(x_out, x, qy, z_mu, log_var, norm=1, recon_N=x_node, node_labels=node_labels, node_multiplier=1, pos_weight=pos_weight, beta_n=1, beta_g=1)
