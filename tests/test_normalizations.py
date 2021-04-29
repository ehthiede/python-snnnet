import pytest
import torch
from snnnet.third_order_layers import _get_third_order_contractions
from snnnet import normalizations


def _build_random_contractions(n, in_channels):
    a3 = torch.randn(3, n, n, n, in_channels)
    a2, a1, a0 = _get_third_order_contractions(a3, dim1=1)
    return a3, a2, a1, a0


def permute_contractions(a3, a2, a1, pidx):
    a3_perm = a3[:, pidx][:, :, pidx][:, :, :, pidx]
    a2_perm = a2[:, pidx][:, :, pidx]
    a1_perm = a1[:, pidx]
    return a3_perm, a2_perm, a1_perm


class TestBatchNorm(object):
    @pytest.mark.parametrize('n', [4, 10])
    @pytest.mark.parametrize('in_channels', [2, 5])
    def test_batch_norm_1d(self, n, in_channels):
        x = torch.randn(3, n, in_channels)
        pidx = torch.randperm(n)  # random permutation
        x_perm = x[:, pidx]

        BN_layer = normalizations.SnBatchNorm1d(in_channels)
        x_norm = BN_layer(x)
        x_norm_perm = x_norm[:, pidx]

        x_perm_norm = BN_layer(x_perm)
        assert(torch.norm(x_perm_norm - x_norm_perm) < 1e-4)

    @pytest.mark.parametrize('n', [4, 10])
    @pytest.mark.parametrize('in_channels', [2, 5])
    def test_batch_norm_2d(self, n, in_channels):
        x = torch.randn(3, n, n, in_channels)
        pidx = torch.randperm(n)  # random permutation
        x_perm = x[:, pidx][:, :, pidx]

        BN_layer = normalizations.SnBatchNorm2d(in_channels)
        x_norm = BN_layer(x)
        x_norm_perm = x_norm[:, pidx][:, :, pidx]

        x_perm_norm = BN_layer(x_perm)
        assert(torch.norm(x_perm_norm - x_norm_perm) < 1e-4)

    @pytest.mark.parametrize('n', [4, 10])
    @pytest.mark.parametrize('in_channels', [2, 5])
    def test_batch_norm_3d(self, n, in_channels):
        x = torch.randn(3, n, n, n, in_channels)
        pidx = torch.randperm(n)  # random permutation
        x_perm = x[:, pidx][:, :, pidx][:, :, :, pidx]

        BN_layer = normalizations.SnBatchNorm3d(in_channels)
        x_norm = BN_layer(x)
        x_norm_perm = x_norm[:, pidx][:, :, pidx][:, :, :, pidx]

        x_perm_norm = BN_layer(x_perm)
        assert(torch.norm(x_perm_norm - x_norm_perm) < 1e-4)


class TestInstanceNorm(object):
    @pytest.mark.parametrize('n', [4, 10])
    @pytest.mark.parametrize('in_channels', [2, 5])
    def test_batch_norm_1d(self, n, in_channels):
        x = torch.randn(3, n, in_channels)
        pidx = torch.randperm(n)  # random permutation
        x_perm = x[:, pidx]

        BN_layer = normalizations.SnInstanceNorm1d(in_channels)
        x_norm = BN_layer(x)
        x_norm_perm = x_norm[:, pidx]

        x_perm_norm = BN_layer(x_perm)
        assert(torch.norm(x_perm_norm - x_norm_perm) < 1e-4)

    @pytest.mark.parametrize('n', [4, 10])
    @pytest.mark.parametrize('in_channels', [2, 5])
    def test_batch_norm_2d(self, n, in_channels):
        x = torch.randn(3, n, n, in_channels)
        pidx = torch.randperm(n)  # random permutation
        x_perm = x[:, pidx][:, :, pidx]

        BN_layer = normalizations.SnInstanceNorm2d(in_channels)
        x_norm = BN_layer(x)
        x_norm_perm = x_norm[:, pidx][:, :, pidx]

        x_perm_norm = BN_layer(x_perm)
        assert(torch.norm(x_perm_norm - x_norm_perm) < 1e-4)

    @pytest.mark.parametrize('n', [4, 10])
    @pytest.mark.parametrize('in_channels', [2, 5])
    def test_batch_norm_3d(self, n, in_channels):
        x = torch.randn(3, n, n, n, in_channels)
        pidx = torch.randperm(n)  # random permutation
        x_perm = x[:, pidx][:, :, pidx][:, :, :, pidx]

        BN_layer = normalizations.SnInstanceNorm3d(in_channels)
        x_norm = BN_layer(x)
        x_norm_perm = x_norm[:, pidx][:, :, pidx][:, :, :, pidx]

        x_perm_norm = BN_layer(x_perm)
        assert(torch.norm(x_perm_norm - x_norm_perm) < 1e-4)
