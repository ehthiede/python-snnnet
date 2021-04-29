import pytest
import torch
from snnnet.third_order_models import SnEncoder3, SnDecoder3


class TestEncoder3(object):
    @pytest.mark.parametrize('channel_list', [[3, 2], [2, 2, 2]])
    @pytest.mark.parametrize('node_channels', [0, 1, 3])
    @pytest.mark.parametrize('expand_mode', ["subdivide", "individual", "all"])
    @pytest.mark.parametrize('lead_in', [None, 5])
    def test_equivariance(self, channel_list, node_channels, expand_mode, lead_in):
        n = 3
        pidx = torch.randperm(n)  # random permutation
        x = torch.randn(4, n, n, 2)
        x_perm = x[:, pidx][:, :, pidx]
        if node_channels > 0:
            x_node = torch.randn(4, n, node_channels)
            x_node_perm = x_node[:, pidx]
        else:
            x_node = x_node_perm = None

        snc = SnEncoder3(2, channel_list, 3, in_node_channels=node_channels, expand_mode=expand_mode, lead_in=lead_in)

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
        snc = SnEncoder3(in_channels, (3, 2), 3)

        # Permute the output
        x_out = snc(x)
        expected_shape = (batch_size, n, 3)
        assert(x_out.shape == expected_shape)

    @pytest.mark.parametrize('node_channels', [0, 1])
    @pytest.mark.parametrize('expand_mode', ["subdivide", "individual"])
    def test_third_order_input(self, node_channels, expand_mode):
        channel_list = [3, 3]
        n = 3
        pidx = torch.randperm(n)  # random permutation
        x = torch.randn(4, n, n, n, 2)
        x_perm = x[:, pidx][:, :, pidx][:, :, :, pidx]
        if node_channels > 0:
            x_node = torch.randn(4, n, node_channels)
            x_node_perm = x_node[:, pidx]
        else:
            x_node = x_node_perm = None

        snc = SnEncoder3(2, channel_list, 3, in_node_channels=node_channels, expand_mode=expand_mode, input_is_o2=False)

        # Permute the output
        x_out = snc(x, x_node=x_node)
        x_out_perm = x_out[:, pidx]

        # Permute the input
        x_perm_out = snc(x_perm, x_node=x_node_perm)
        assert(torch.norm(x_out_perm - x_perm_out) < 1E-3)


class TestDecoder3(object):
    @pytest.mark.parametrize('channel_list', [[3, 2], [2]])
    @pytest.mark.parametrize('outer_type', ["all", "individual"])
    @pytest.mark.parametrize('node_channels', [0, 1, 3])
    @pytest.mark.parametrize('expand_mode', ["subdivide", "individual", 'direct'])
    @pytest.mark.parametrize('lead_in', [None, 5])
    def test_equivariance(self, channel_list, outer_type, node_channels, expand_mode, lead_in):
        n = 3
        pidx = torch.randperm(n)  # random permutation
        x = torch.randn(4, n, 2)
        x_perm = x[:, pidx]

        snc = SnDecoder3(2, channel_list, 3, out_node_channels=node_channels,
                         outer_type=outer_type, expand_mode=expand_mode, lead_in=lead_in)

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
        snc = SnDecoder3(in_channels, (3,), 3, outer_type=outer_type)

        # )

        # Permute the output
        x_out = snc(x)
        expected_shape = (batch_size, n, n, 3)
        assert(x_out.shape == expected_shape)

    @pytest.mark.parametrize('node_channels', [0, 1])
    @pytest.mark.parametrize('expand_mode', ["subdivide", "individual"])
    def test_third_order_output(self, node_channels, expand_mode):
        channel_list = [3, 3]
        n = 3
        pidx = torch.randperm(n)  # random permutation
        x = torch.randn(4, n, 2)
        x_perm = x[:, pidx]

        snc = SnDecoder3(2, channel_list, 3, out_node_channels=node_channels,
                         expand_mode=expand_mode, output_is_o2=False)

        # Permute the output
        if node_channels > 0:
            x_out, x_node_out = snc(x)
            x_node_out_perm = x_node_out[:, pidx]
            x_perm_out, x_perm_node_out = snc(x_perm)
        else:
            x_out = snc(x)
            x_perm_out = snc(x_perm)
        x_out_perm = x_out[:, pidx][:, :, pidx][:, :, :, pidx]

        # Permute the input
        assert(torch.norm(x_out_perm - x_perm_out) < 1E-3)

        if node_channels > 0:
            assert(torch.norm(x_node_out_perm - x_perm_node_out) < 1E-3)
