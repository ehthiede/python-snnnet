import pytest
import torch
from snnnet.second_order_layers import SnConv2, SnConv2to1


def _build_mask(batch_size, n, dim=2):
    mask = torch.randint(0, 2, (batch_size, n)).bool()
    mask[:, 0] = True
    if dim == 1:
        return mask
    if dim == 2:
        mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        return mask
    elif dim == 3:
        new_mask = mask.unsqueeze(1)
        new_mask = new_mask * mask.unsqueeze(2)
        new_mask = new_mask.unsqueeze(3)
        new_mask = new_mask * mask.unsqueeze(1).unsqueeze(2)
        return new_mask
    else:
        raise Exception("Dim not recognized.  This routine was coded poorly, so this is really on us.")


class TestSnConv2(object):
    @pytest.mark.parametrize('n', [4, 10])
    @pytest.mark.parametrize('in_channels', [2, 5])
    @pytest.mark.parametrize('use_mask', [True, False])
    @pytest.mark.parametrize('node_channels', [0, 1, 3])
    @pytest.mark.parametrize('diagonal_free', [True, False])
    @pytest.mark.parametrize('inner_bias', [True, False])
    def test_equivariance(self, n, in_channels, use_mask, node_channels, diagonal_free, inner_bias):
        x = torch.randn(4, n, n, in_channels)
        pidx = torch.randperm(n)  # random permutation
        if use_mask:
            mask = _build_mask(4, n, 2)
            x[~mask] = 0
        else:
            mask = None
        if node_channels > 0:
            x_node = torch.randn(4, n, node_channels)
            x_node_perm = x_node[:, pidx]
        else:
            x_node = x_node_perm = None

        snc = SnConv2(in_channels, 2, node_channels, diagonal_free=diagonal_free, inner_bias=inner_bias)

        # Permute the output
        x.requires_grad = True
        x_out = snc(x, x_node=x_node, mask=mask)
        x_out_perm = x_out[:, pidx][:, :, pidx]

        # Permute the input
        x_perm = x[:, pidx][:, :, pidx]
        if use_mask:
            perm_mask = mask[:, pidx][:, :, pidx]
        else:
            perm_mask = None
        x_perm_out = snc(x_perm, x_node=x_node_perm, mask=perm_mask)
        assert(torch.norm(x_out_perm - x_perm_out) < 1E-4)

    @pytest.mark.parametrize('batch_size', [1, 3])
    @pytest.mark.parametrize('n', [1, 3])
    @pytest.mark.parametrize('in_channels', [1, 3])
    @pytest.mark.parametrize('out_channels', [1, 3])
    def test_output_shape(self, batch_size, n, in_channels, out_channels):
        x = torch.randn(batch_size, n, n, in_channels)
        snc = SnConv2(in_channels, out_channels)

        # Permute the output
        x_out = snc(x)
        assert(x_out.shape[-1] == out_channels)

    @pytest.mark.parametrize('batch_size', [1, 3])
    @pytest.mark.parametrize('n', [1, 3])
    @pytest.mark.parametrize('in_channels', [1, 3])
    @pytest.mark.parametrize('out_channels', [1, 3])
    def test_masking(self, batch_size, n, in_channels, out_channels):
        x = torch.randn(batch_size, n, n, in_channels)
        mask = _build_mask(batch_size, n, 2)
        x[~mask] = 0
        snc = SnConv2(in_channels, out_channels)
        x_out = snc(x, mask=mask)
        assert(torch.norm(x_out[~mask]) == 0)


class TestSnConv2to1(object):
    @pytest.mark.parametrize('n', [4, 10])
    @pytest.mark.parametrize('in_channels', [2, 5])
    @pytest.mark.parametrize('use_mask', [True, False])
    @pytest.mark.parametrize('inner_bias', [True, False])
    def test_equivariance(self, n, in_channels, use_mask, inner_bias):
        x = torch.randn(4, n, n, in_channels)
        pidx = torch.randperm(n)  # random permutation
        if use_mask:
            mask = _build_mask(4, n, dim=2)
            x[~mask] = 0
        else:
            mask = None

        snc = SnConv2to1(in_channels, 2, inner_bias=inner_bias)

        # Permute the output
        x_out = snc(x, mask=mask)
        x_out_perm = x_out[:, pidx]

        # Permute the input
        x_perm = x[:, pidx][:, :, pidx]
        if use_mask:
            perm_mask = mask[:, pidx][:, :, pidx]
        else:
            perm_mask = None
        x_perm_out = snc(x_perm, mask=perm_mask)
        assert(torch.norm(x_out_perm - x_perm_out) < 1E-4)

    @pytest.mark.parametrize('batch_size', [1, 3])
    @pytest.mark.parametrize('n', [1, 3])
    @pytest.mark.parametrize('in_channels', [1, 3])
    @pytest.mark.parametrize('out_channels', [1, 3])
    def test_output_shape(self, batch_size, n, in_channels, out_channels):
        x = torch.randn(batch_size, n, n, in_channels)
        snc = SnConv2to1(in_channels, out_channels)

        # Permute the output
        x_out = snc(x)
        assert(x_out.shape[-1] == out_channels)

    @pytest.mark.parametrize('in_channels', [1, 3])
    @pytest.mark.parametrize('out_channels', [1, 3])
    def test_masking(self, in_channels, out_channels):
        n = 4
        x = torch.randn(3, n, n, in_channels)
        mask = torch.ones(x.shape[:-1]).bool()
        mask[0, (n-1):, :] = False
        mask[0, :, (n-1):] = False
        mask[1, (n-2):, :] = False
        mask[1, :, (n-2):] = False
        mask_1d = mask[:, :, 0]
        x[~mask] = 0
        snc = SnConv2to1(in_channels, out_channels)
        x_out = snc(x, mask=mask)
        assert(torch.norm(x_out[~mask_1d]) == 0)
