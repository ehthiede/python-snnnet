import pytest
import torch
from snnnet.second_order_models import SnEncoder, SnDecoder, SnVAE, SnLocalEncoder, SnLocalDecoder


class TestEncoder(object):
    @pytest.mark.parametrize('channel_list', [(2,), [7, 5], []])
    @pytest.mark.parametrize('node_channels', [0, 1, 3])
    @pytest.mark.parametrize('diagonal_free', [True, False])
    def test_equivariance(self, channel_list, node_channels, diagonal_free):
        n = 3
        pidx = torch.randperm(n)  # random permutation
        x = torch.randn(4, n, n, 2)
        x_perm = x[:, pidx][:, :, pidx]
        if node_channels > 0:
            x_node = torch.randn(4, n, node_channels)
            x_node_perm = x_node[:, pidx]
        else:
            x_node = x_node_perm = None

        snc = SnEncoder(2, channel_list, 3, in_node_channels=node_channels, diagonal_free=diagonal_free)

        # Permute the output
        x_out = snc(x, x_node=x_node)
        x_out_perm = x_out[:, pidx]

        # Permute the input
        x_perm_out = snc(x_perm, x_node=x_node_perm)
        assert(torch.norm(x_out_perm - x_perm_out) < 1E-3)

    @pytest.mark.parametrize('batch_size', [1, 3])
    @pytest.mark.parametrize('n', [1, 3])
    @pytest.mark.parametrize('in_channels', [1, 3])
    @pytest.mark.parametrize('out_channels', [1, 3])
    def test_output_shape(self, batch_size, n, in_channels, out_channels):
        x = torch.randn(batch_size, n, n, in_channels)
        snc = SnEncoder(in_channels, (3,), 3)

        # Permute the output
        x_out = snc(x)
        expected_shape = (batch_size, n, 3)
        assert(x_out.shape == expected_shape)

    @pytest.mark.parametrize('in_channels', [1, 3])
    @pytest.mark.parametrize('out_channels', [1, 3])
    def test_masking(self, in_channels, out_channels):
        n = 4
        x = torch.randn(3, n, n, in_channels)
        mask = torch.ones(x.shape[:-1]).bool()
        mask[0, (n-1):, (n-1):] = 0
        mask[1, (n-2):, (n-2):] = 0
        mask_1d = mask[:, :, 0]
        x[~mask] = 0
        snc = SnEncoder(in_channels, [], 3)
        x_out = snc(x, mask=mask)
        assert(torch.norm(x_out[~mask_1d]) == 0)


class TestDecoder(object):
    @pytest.mark.parametrize('channel_list', [(2,), [7, 5]])
    @pytest.mark.parametrize('outer_type', ["all", "individual"])
    @pytest.mark.parametrize('node_channels', [0, 1, 3])
    @pytest.mark.parametrize('diagonal_free', [True, False])
    def test_equivariance(self, channel_list, outer_type, node_channels, diagonal_free):
        n = 3
        pidx = torch.randperm(n)  # random permutation
        x = torch.randn(4, n, 2)
        x_perm = x[:, pidx]

        snc = SnDecoder(2, channel_list, 3, out_node_channels=node_channels,
                        diagonal_free=diagonal_free, outer_type=outer_type)

        # Permute the output
        if node_channels > 0:
            x_out, x_node_out = snc(x)
            x_node_out_perm = x_node_out[:, pidx]
            x_perm_out, x_perm_node_out = snc(x_perm)
        else:
            x_out = snc(x)
            x_perm_out = snc(x_perm)
        x_out_perm = x_out[:, pidx][:, :, pidx]
        assert(torch.norm(x_out_perm - x_perm_out) < 1E-3)

        if node_channels > 0:
            assert(torch.norm(x_node_out_perm - x_perm_node_out) < 1E-3)

    @pytest.mark.parametrize('batch_size', [1, 3])
    @pytest.mark.parametrize('n', [1, 3])
    @pytest.mark.parametrize('in_channels', [1, 3])
    @pytest.mark.parametrize('out_channels', [1, 3])
    @pytest.mark.parametrize('outer_type', ["all", "individual"])
    def test_output_shape(self, batch_size, n, in_channels, out_channels, outer_type):
        x = torch.randn(batch_size, n, in_channels)
        snc = SnDecoder(in_channels, (3,), 3, outer_type=outer_type)

        # )

        # Permute the output
        x_out = snc(x)
        expected_shape = (batch_size, n, n, 3)
        assert(x_out.shape == expected_shape)

    @pytest.mark.parametrize('in_channels', [1, 3])
    @pytest.mark.parametrize('out_channels', [1, 3])
    @pytest.mark.parametrize('outer_type', ["all", "individual"])
    def test_masking(self, in_channels, out_channels, outer_type):
        n = 4
        x = torch.randn(3, n, in_channels)
        snc = SnDecoder(in_channels, [], 3, outer_type=outer_type)
        mask = torch.ones((3, n, n)).bool()
        mask[0, (n-1):, (n-1):] = 0
        mask[1, (n-2):, (n-2):] = 0
        mask_1d = mask[:, :, 0]
        x[~mask_1d] = 0
        x_out = snc(x, mask=mask)
        assert(torch.norm(x_out[~mask]) == 0)


class TestVAE(object):
    @pytest.mark.parametrize('hidden_size', [2, 4])
    @pytest.mark.parametrize('outer_type', ["all", "individual"])
    def test_hidden_equivariance(self, hidden_size, outer_type):
        n = 3
        pidx = torch.randperm(n)  # random permutation
        x = torch.randn(4, n, n, 2)
        pidx = torch.randperm(n)  # random permutation

        encoder = SnEncoder(2, (3,), 2 * hidden_size)
        decoder = SnDecoder(hidden_size, (3,), 2, outer_type=outer_type)
        vae = SnVAE(encoder, decoder)

        # Permute the output
        x_out, z_mu, z_var = vae(x)
        z_mu = z_mu[:, pidx]
        z_var = z_var[:, pidx]

        # Permute the input
        x_perm = x[:, pidx][:, :, pidx]
        x_perm_out, z_mu_perm, z_var_perm = vae(x_perm)
        assert(torch.norm(z_mu - z_mu_perm) < 1E-3)
        assert(torch.norm(z_var - z_var_perm) < 1E-3)

    def raises_error():
        n = 3
        x = torch.randn(4, n, n, 2)

        encoder = SnEncoder(2, (3,), 3)
        decoder = SnDecoder(3, (3,), 2, outer_type='all')
        vae = SnVAE(encoder, decoder)
        with pytest.raises(AssertionError):
            vae(x)

    @pytest.mark.parametrize('batch_size', [1, 3])
    @pytest.mark.parametrize('n', [1, 3])
    @pytest.mark.parametrize('in_channels', [1, 3])
    @pytest.mark.parametrize('outer_type', ["all", "individual"])
    @pytest.mark.parametrize('hidden_size', [2, 4])
    def test_output_shape(self, batch_size, n, in_channels, hidden_size, outer_type):
        x = torch.randn(batch_size, n, n, in_channels)
        encoder = SnEncoder(in_channels, (3,), 2 * hidden_size)
        decoder = SnDecoder(hidden_size, (3,), in_channels, outer_type=outer_type)
        vae = SnVAE(encoder, decoder)
        x_out, __, __ = vae(x)
        assert(x_out.shape == x.shape)


class TestLocalEncoder(object):
    @pytest.mark.parametrize('channel_list', [(2,), [7, 5], []])
    @pytest.mark.parametrize('node_channels', [0, 1, 3])
    @pytest.mark.parametrize('neighbor_channels', [1, 3])
    def test_equivariance(self, channel_list, node_channels, neighbor_channels):
        n = 3
        pidx = torch.randperm(n)  # random permutation
        x = torch.randn(4, n, n, n, 2)
        x_perm = x[:, pidx][:, :, pidx][:, :, :, pidx]
        if node_channels > 0:
            x_node = torch.randn(4, n, n, node_channels)
            x_node_perm = x_node[:, pidx][:, :, pidx]
        else:
            x_node = x_node_perm = None
        nc_list = [neighbor_channels] * len(channel_list)

        snc = SnLocalEncoder(2, channel_list, nc_list, 5, in_node_channels=node_channels)

        # Permute the output
        x_out = snc(x, x_node=x_node)
        print(x_out.shape)
        x_out_perm = x_out[:, pidx]

        # Permute the input
        x_perm_out = snc(x_perm, x_node=x_node_perm)
        assert(torch.norm(x_out_perm - x_perm_out) < 1E-3)

    @pytest.mark.parametrize('in_channels', [1, 3])
    @pytest.mark.parametrize('out_channels', [1, 3])
    @pytest.mark.parametrize('channel_list', [(2,), [7, 5], []])
    @pytest.mark.parametrize('node_channels', [0, 1, 3])
    @pytest.mark.parametrize('neighbor_channels', [1, 3])
    def test_masking(self, in_channels, out_channels, channel_list, node_channels, neighbor_channels):
        n = 4
        x = torch.randn(4, n, n, n, in_channels)
        mask = torch.ones(x.shape[:-1]).bool()
        mask[0, (n-1):, (n-1):, (n-1)] = 0
        mask[1, (n-2):, (n-2):, (n-1)] = 0
        if node_channels > 0:
            x_node = torch.randn(4, n, n, node_channels)
            x_node[~mask[:, :, :, 0]] = 0.
        else:
            x_node = None
        mask_1d = mask[:, :, 0, 0]
        x[~mask] = 0
        nc_list = [neighbor_channels] * len(channel_list)
        snc = SnLocalEncoder(in_channels, channel_list, nc_list, out_channels, in_node_channels=node_channels)
        x_out = snc(x, x_node=x_node, mask=mask)
        assert(torch.norm(x_out[~mask_1d]) == 0)


class TestLocalDecoder(object):
    @pytest.mark.parametrize('channel_list', [(2,), [7, 5], []])
    # @pytest.mark.parametrize('outer_type', ["all", "individual"])
    @pytest.mark.parametrize('outer_type', ["individual"])
    @pytest.mark.parametrize('node_channels', [0, 1, 3])
    @pytest.mark.parametrize('neighbor_channels', [1, 3])
    def test_equivariance(self, channel_list, outer_type, node_channels, neighbor_channels):
        n = 3
        pidx = torch.randperm(n)  # random permutation
        x = torch.randn(4, n, 2)
        x_perm = x[:, pidx]
        nc_list = [neighbor_channels] * len(channel_list)

        snc = SnLocalDecoder(2, channel_list, nc_list, 3, out_node_channels=node_channels,
                             outer_type=outer_type)

        # Permute the output
        if node_channels > 0:
            x_out, x_node_out = snc(x)
            x_node_out_perm = x_node_out[:, pidx][:, :, pidx]
            x_perm_out, x_perm_node_out = snc(x_perm)
        else:
            x_out = snc(x)
            x_perm_out = snc(x_perm)
        x_out_perm = x_out[:, pidx][:, :, pidx][:, :, :, pidx]
        assert(torch.norm(x_out_perm - x_perm_out) < 1E-3)

        if node_channels > 0:
            assert(torch.norm(x_node_out_perm - x_perm_node_out) < 1E-3)

    @pytest.mark.parametrize('in_channels', [1, 3])
    @pytest.mark.parametrize('out_channels', [1, 3])
    @pytest.mark.parametrize('outer_type', ["individual"])
    def test_masking(self, in_channels, out_channels, outer_type):
        n = 4
        x = torch.randn(3, n, in_channels)
        snc = SnLocalDecoder(in_channels, [2, 2], [3, 1], 3, outer_type=outer_type)
        mask = torch.ones((3, n, n, n)).bool()
        mask[0, (n-1):, (n-1):, (n-1)] = 0
        mask[1, (n-2):, (n-2):, (n-1)] = 0
        mask_1d = mask[:, :, 0, 0]
        x[~mask_1d] = 0
        x_out = snc(x, mask=mask)
        assert(torch.norm(x_out[~mask]) == 0)
