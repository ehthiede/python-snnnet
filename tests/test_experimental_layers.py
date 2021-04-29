import pytest
import torch
from snnnet.experimental_layers import GraphConv1D, GraphConv2D, ResidualBlockO2, ResidualBlockO3


@pytest.mark.parametrize('hidden_channels', [None, 2, 5])
@pytest.mark.parametrize('batch_norm', [True, False])
def test_o2residual_equivariance(hidden_channels, batch_norm):
    n = 5
    in_channels = 4
    x = torch.randn(4, n, n, in_channels)
    pidx = torch.randperm(n)  # random permutation

    block = ResidualBlockO2(in_channels, hidden_channels, batch_norm=batch_norm)
    x_out = block(x)
    x_out_perm = x_out[:, pidx][:, :, pidx]

    x_perm = x[:, pidx][:, :, pidx]
    x_perm_out = block(x_perm)
    assert(torch.allclose(x_out_perm, x_perm_out, atol=1e-6, rtol=1e-4))


@pytest.mark.parametrize('hidden_channels', [None, 2, 5])
@pytest.mark.parametrize('batch_norm', [True, False])
def test_o3residual_equivariance(hidden_channels, batch_norm):
    n = 5
    in_channels = 4
    x = torch.randn(4, n, n, n, in_channels)
    pidx = torch.randperm(n)  # random permutation

    block = ResidualBlockO3(in_channels, hidden_channels, batch_norm=batch_norm)
    x_out = block(x)
    x_out_perm = x_out[:, pidx][:, :, pidx][:, :, :, pidx]

    x_perm = x[:, pidx][:, :, pidx][:, :, :, pidx]
    x_perm_out = block(x_perm)
    assert(torch.allclose(x_out_perm, x_perm_out, atol=1e-6, rtol=1e-4))


class TestGraphConv1D(object):
    @pytest.mark.parametrize('n', [4, 10])
    @pytest.mark.parametrize('in_channels', [2, 5])
    def test_equivariance_1d(self, n, in_channels):
        x = torch.randn(11, n, in_channels)
        adj_mat = torch.randint(0, 2, (11, n, n)).float()
        pidx = torch.randperm(n)  # random permutation

        snc = GraphConv1D(in_channels, 2)

        # Permute the output
        x_out = snc(x, adj_mat)
        x_out_perm = x_out[:, pidx]

        # Permute the input
        adj_mat_perm = adj_mat[:, pidx][:, :, pidx]
        x_perm = x[:, pidx]
        x_perm_out = snc(x_perm, adj_mat_perm)
        assert(torch.norm(x_out_perm - x_perm_out) < 1E-4)


class TestGraphConv2D(object):
    @pytest.mark.parametrize('n', [4, 10])
    @pytest.mark.parametrize('in_channels', [2, 5])
    def test_equivariance_1d(self, n, in_channels):
        x = torch.randn(11, n, n, in_channels)
        adj_mat = torch.randint(0, 2, (11, n, n)).float()
        pidx = torch.randperm(n)  # random permutation

        snc = GraphConv2D(in_channels, 2)

        # Permute the output
        x_out = snc(x, adj_mat)
        x_out_perm = x_out[:, pidx]
        # Permute the input
        adj_mat_perm = adj_mat[:, pidx][:, :, pidx]
        x_perm = x[:, pidx]
        x_perm_out = snc(x_perm, adj_mat_perm)
        assert(torch.norm(x_out_perm - x_perm_out) < 1E-4)
