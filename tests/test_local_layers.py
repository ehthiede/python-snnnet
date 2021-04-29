import pytest
import torch
from snnnet.second_order_layers import LocalConv2, Neighborhood
from test_second_order_layers import _build_mask


class TestLocalConv2(object):
    @pytest.mark.parametrize('n', [2, 4])
    @pytest.mark.parametrize('input_channels', [1, 5])
    @pytest.mark.parametrize('neighb_channels', [1, 5])
    @pytest.mark.parametrize('output_channels', [1, 5])
    @pytest.mark.parametrize('use_mask', [True, False])
    @pytest.mark.parametrize('node_channels', [0, 1, 3])
    @pytest.mark.parametrize('product_type', ['matmul', 'elementwise'])
    def test_equivariance(self, n, input_channels, neighb_channels, output_channels, node_channels, use_mask, product_type):
        x = torch.randn(3, n, n, n, input_channels)
        pidx = torch.randperm(n)  # random permutation
        if use_mask:
            mask = _build_mask(3, n, dim=3)
            x[~mask] = 0
        else:
            mask = None

        model = LocalConv2(input_channels, neighb_channels, output_channels, pdim1=1, product_type=product_type)

        # Permute the output
        x_out = model(x, mask=mask)
        x_out_perm = x_out[:, pidx][:, :, pidx][:, :, :, pidx]

        # Permute the input
        x_perm = x[:, pidx][:, :, pidx][:, :, :, pidx]
        if use_mask:
            perm_mask = mask[:, pidx][:, :, pidx][:, :, :, pidx]
        else:
            perm_mask = None
        x_perm_out = model(x_perm, mask=perm_mask)
        assert(torch.norm(x_out_perm - x_perm_out) < 1E-4)


class TestNeighborhood(object):
    @pytest.mark.parametrize('n', [2, 4])
    @pytest.mark.parametrize('input_channels', [1, 5])
    @pytest.mark.parametrize('neighb_channels', [1, 5])
    @pytest.mark.parametrize('output_channels', [1, 5])
    @pytest.mark.parametrize('use_mask', [True, False])
    def test_equivariance(self, n, input_channels, neighb_channels, output_channels, use_mask):
        x = torch.randn(3, n, n, n, input_channels)
        pidx = torch.randperm(n)  # random permutation
        if use_mask:
            mask = _build_mask(3, n, dim=3)
            x[~mask] = 0
        else:
            mask = None

        model = Neighborhood(input_channels, neighb_channels, pdim1=1)

        # Permute the output
        x_out = model(x, mask=mask)
        x_out_perm = x_out[:, pidx][:, :, pidx]

        # Permute the input
        x_perm = x[:, pidx][:, :, pidx][:, :, :, pidx]
        if use_mask:
            perm_mask = mask[:, pidx][:, :, pidx][:, :, :, pidx]
        else:
            perm_mask = None
        x_perm_out = model(x_perm, mask=perm_mask)
        assert(torch.norm(x_out_perm - x_perm_out) < 1E-4)
