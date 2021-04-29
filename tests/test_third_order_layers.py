import pytest
import torch
from snnnet.third_order_layers import _get_third_order_contractions, SnConv3, SnConv3to2, SnConv3to1, expand_1_to_3


class Test3OContractions(object):
    @pytest.mark.parametrize('n', [4, 10])
    @pytest.mark.parametrize('in_channels', [1, 2, 5])
    def test_equivariance(self, n, in_channels):
        x = torch.randn(1, n, n, n, in_channels)

        pidx = torch.randperm(n)  # random permutation
        x_perm = x[:, pidx][:, :, pidx][:, :, :, pidx]
        contractions = _get_third_order_contractions(x, dim1=1)
        perm_contractions = _get_third_order_contractions(x_perm, dim1=1)

        # Check third_order_contraction
        x2_out_perm = contractions[0][:, pidx][:, :, pidx]
        x1_out_perm = contractions[1][:, pidx]
        x0_out_perm = contractions[2]

        assert(torch.norm(x2_out_perm - perm_contractions[0]) < 1e-4)
        assert(torch.norm(x1_out_perm - perm_contractions[1]) < 1e-4)
        assert(torch.norm(x0_out_perm - perm_contractions[2]) < 1e-4)


class TestSnConv3(object):
    @pytest.mark.parametrize('n', [4, 10])
    @pytest.mark.parametrize('in_channels', [1, 2, 5])
    @pytest.mark.parametrize('out_channels', [1, 2, 5])
    def test_mix_third_order_equivariance(self, n, in_channels, out_channels):
        a3 = torch.randn(2, n, n, n, in_channels)
        a2 = torch.randn(2, n, n, 6 * in_channels)
        a1 = torch.randn(2, n, 10 * in_channels)
        a0 = torch.randn(2, 5 * in_channels)

        pidx = torch.randperm(n)  # random permutation
        a3_perm = a3[:, pidx][:, :, pidx][:, :, :, pidx]
        a2_perm = a2[:, pidx][:, :, pidx]
        a1_perm = a1[:, pidx]
        a0_perm = a0

        snc = SnConv3(in_channels, out_channels)
        third_order_mix = snc.mix_third_order(a3, a2, a1, a0)
        third_order_perm_mix = snc.mix_third_order(a3_perm, a2_perm, a1_perm, a0_perm)

        third_order_mix_perm = third_order_mix[:, pidx][:, :, pidx][:, :, :, pidx]
        assert(torch.norm(third_order_mix_perm - third_order_perm_mix) < 1e-4)

    @pytest.mark.parametrize('n', [4, 10])
    @pytest.mark.parametrize('in_channels', [1, 2, 5])
    @pytest.mark.parametrize('out_channels', [1, 2, 5])
    def test_mix_second_order_equivariance(self, n, in_channels, out_channels):
        a2 = torch.randn(2, n, n, 6 * in_channels)
        a1 = torch.randn(2, n, 10 * in_channels)
        a0 = torch.randn(2, 5 * in_channels)

        pidx = torch.randperm(n)  # random permutation
        a2_perm = a2[:, pidx][:, :, pidx]
        a1_perm = a1[:, pidx]
        a0_perm = a0

        snc = SnConv3(in_channels, out_channels)
        second_order_mixes = snc.mix_second_order(a2, a1, a0)
        second_order_perm_mixes = snc.mix_second_order(a2_perm, a1_perm, a0_perm)

        for som_i, sopm_i in zip(second_order_mixes, second_order_perm_mixes):
            som_i_perm = som_i[:, pidx][:, :, pidx]
            assert(torch.norm(som_i_perm - sopm_i) < 1e-4)

    @pytest.mark.parametrize('n', [4, 10])
    @pytest.mark.parametrize('in_channels', [1, 2, 5])
    @pytest.mark.parametrize('out_channels', [1, 2, 5])
    def test_mix_first_order_equivariance(self, n, in_channels, out_channels):
        a1 = torch.randn(2, n, 10 * in_channels)
        a0 = torch.randn(2, 5 * in_channels)

        pidx = torch.randperm(n)  # random permutation
        a1_perm = a1[:, pidx]
        a0_perm = a0

        snc = SnConv3(in_channels, out_channels)
        first_order_mix = snc.mix_first_order(a1, a0)
        first_order_perm_mix = snc.mix_first_order(a1_perm, a0_perm)

        first_order_mix_perm = first_order_mix[:, pidx]
        assert(torch.norm(first_order_mix_perm - first_order_perm_mix) < 1e-4)

    @pytest.mark.parametrize('n', [4, 10])
    @pytest.mark.parametrize('in_channels', [1, 2, 5])
    @pytest.mark.parametrize('out_channels', [1, 2, 5])
    @pytest.mark.parametrize('in_node_channels', [0, 3])
    def test_full_equivariance(self, n, in_channels, out_channels, in_node_channels):
        x = torch.randn(2, n, n, n, in_channels)
        if in_node_channels > 0:
            x_node = torch.randn(2, n, in_node_channels)
        else:
            x_node = None

        pidx = torch.randperm(n)  # random permutation
        x_perm = x[:, pidx][:, :, pidx][:, :, :, pidx]
        x_node_perm = x_node
        if x_node_perm is not None:
            x_node_perm = x_node[:, pidx]

        snc = SnConv3(in_channels, out_channels, in_node_channels=in_node_channels)
        x_out = snc(x, x_node=x_node)
        x_out_perm = x_out[:, pidx][:, :, pidx][:, :, :, pidx]

        x_perm_out = snc(x_perm, x_node_perm)
        assert(torch.norm(x_out_perm - x_perm_out) < 1e-4)

    @pytest.mark.parametrize('batch_size', [2, 3])
    @pytest.mark.parametrize('contraction_norm', ['batch', 'instance'])
    @pytest.mark.parametrize('mixture_norm', ['batch', 'instance'])
    def test_norm_equivariance(self, batch_size, contraction_norm, mixture_norm):
        n = 4
        in_channels = out_channels = 5
        x = torch.randn(batch_size, n, n, n, in_channels)
        pidx = torch.randperm(n)  # random permutation
        x_perm = x[:, pidx][:, :, pidx][:, :, :, pidx]

        snc = SnConv3(in_channels, out_channels, contraction_norm=contraction_norm, mixture_norm=mixture_norm)
        x_out = snc(x)
        x_out_perm = x_out[:, pidx][:, :, pidx][:, :, :, pidx]

        x_perm_out = snc(x_perm)
        assert(torch.norm(x_out_perm - x_perm_out) < 1e-4)


class TestSnConv3to2(object):
    @pytest.mark.parametrize('n', [4, 10])
    @pytest.mark.parametrize('in_channels', [1, 2, 5])
    @pytest.mark.parametrize('out_channels', [1, 2, 5])
    def test_mix_second_order_equivariance(self, n, in_channels, out_channels):
        a2 = torch.randn(2, n, n, 6 * in_channels)
        a1 = torch.randn(2, n, 10 * in_channels)
        a0 = torch.randn(2, 5 * in_channels)

        pidx = torch.randperm(n)  # random permutation
        a2_perm = a2[:, pidx][:, :, pidx]
        a1_perm = a1[:, pidx]
        a0_perm = a0

        snc = SnConv3to2(in_channels, out_channels)
        som_i = snc.mix_second_order(a2, a1, a0)
        sopm_i = snc.mix_second_order(a2_perm, a1_perm, a0_perm)
        som_i_perm = som_i[:, pidx][:, :, pidx]
        assert(torch.norm(som_i_perm - sopm_i) < 1e-4)

    @pytest.mark.parametrize('n', [4, 10])
    @pytest.mark.parametrize('in_channels', [1, 2, 5])
    @pytest.mark.parametrize('out_channels', [1, 2, 5])
    def test_mix_first_order_equivariance(self, n, in_channels, out_channels):
        a1 = torch.randn(2, n, 10 * in_channels)
        a0 = torch.randn(2, 5 * in_channels)

        pidx = torch.randperm(n)  # random permutation
        a1_perm = a1[:, pidx]
        a0_perm = a0

        snc = SnConv3to2(in_channels, out_channels)
        first_order_mix = snc.mix_first_order(a1, a0)
        first_order_perm_mix = snc.mix_first_order(a1_perm, a0_perm)

        first_order_mix_perm = first_order_mix[:, pidx]
        assert(torch.norm(first_order_mix_perm - first_order_perm_mix) < 1e-4)

    @pytest.mark.parametrize('n', [4, 10])
    @pytest.mark.parametrize('in_channels', [1, 2, 5])
    @pytest.mark.parametrize('out_channels', [1, 2, 5])
    def test_full_equivariance(self, n, in_channels, out_channels):
        x = torch.randn(2, n, n, n, in_channels)
        pidx = torch.randperm(n)  # random permutation
        x_perm = x[:, pidx][:, :, pidx][:, :, :, pidx]

        snc = SnConv3to2(in_channels, out_channels)
        x_out = snc(x)
        x_out_perm = x_out[:, pidx][:, :, pidx]

        x_perm_out = snc(x_perm)
        assert(torch.norm(x_out_perm - x_perm_out) < 1e-4)

    @pytest.mark.parametrize('batch_size', [2, 3])
    @pytest.mark.parametrize('contraction_norm', ['batch', 'instance'])
    @pytest.mark.parametrize('mixture_norm', ['batch', 'instance'])
    def test_batch_equivariance(self, batch_size, contraction_norm, mixture_norm):
        n = 4
        in_channels = out_channels = 5
        x = torch.randn(batch_size, n, n, n, in_channels)
        pidx = torch.randperm(n)  # random permutation
        x_perm = x[:, pidx][:, :, pidx][:, :, :, pidx]

        snc = SnConv3to2(in_channels, out_channels, contraction_norm=contraction_norm, mixture_norm=mixture_norm)
        x_out = snc(x)
        x_out_perm = x_out[:, pidx][:, :, pidx]

        x_perm_out = snc(x_perm)
        assert(torch.norm(x_out_perm - x_perm_out) < 1e-4)


class TestSnConv3to1(object):
    @pytest.mark.parametrize('n', [4, 10])
    @pytest.mark.parametrize('in_channels', [1, 2, 5])
    @pytest.mark.parametrize('out_channels', [1, 2, 5])
    def test_full_equivariance(self, n, in_channels, out_channels):
        x = torch.randn(2, n, n, n, in_channels)
        pidx = torch.randperm(n)  # random permutation
        x_perm = x[:, pidx][:, :, pidx][:, :, :, pidx]

        snc = SnConv3to1(in_channels, out_channels)
        x_out = snc(x)
        x_out_perm = x_out[:, pidx]

        x_perm_out = snc(x_perm)
        assert(torch.norm(x_out_perm - x_perm_out) < 1e-4)

    @pytest.mark.parametrize('batch_size', [2, 3])
    @pytest.mark.parametrize('contraction_norm', ['batch', 'instance'])
    @pytest.mark.parametrize('mixture_norm', ['batch', 'instance'])
    def test_batch_equivariance(self, batch_size, contraction_norm, mixture_norm):
        n = 4
        in_channels = out_channels = 5
        x = torch.randn(batch_size, n, n, n, in_channels)
        pidx = torch.randperm(n)  # random permutation
        x_perm = x[:, pidx][:, :, pidx][:, :, :, pidx]

        snc = SnConv3to1(in_channels, out_channels, contraction_norm=contraction_norm, mixture_norm=mixture_norm)
        x_out = snc(x)
        x_out_perm = x_out[:, pidx]

        x_perm_out = snc(x_perm)
        assert(torch.norm(x_out_perm - x_perm_out) < 1e-4)


class TestExpand1to3(object):
    @pytest.mark.parametrize('n', [4, 10])
    @pytest.mark.parametrize('in_channels', [1, 2, 5])
    @pytest.mark.parametrize('mode', ['direct', 'linear'])
    def test_batch_equivariance(self, n, in_channels, mode):
        x = torch.randn(3, n, in_channels)
        pidx = torch.randperm(n)  # random permutation
        x_perm = x[:, pidx]
        x_exp = expand_1_to_3(x, mode=mode)

        x_exp_perm = x_exp[:, pidx][:, :, pidx][:, :, :, pidx]
        x_perm_exp = expand_1_to_3(x_perm, mode=mode)

        assert(torch.norm(x_exp_perm - x_perm_exp) < 1e-4)

    @pytest.mark.parametrize('n', [4, 10])
    @pytest.mark.parametrize('in_channels', [1, 2, 5])
    def test_direct_shape(self, n, in_channels):
        x = torch.randn(4, n, in_channels)
        x_exp = expand_1_to_3(x)
        correct_shape = (4, n, n, n, 3 * in_channels)
        for i in range(5):
            assert(x_exp.shape[i] == correct_shape[i])

    def test_linear_values(self):
        n = 3
        x = torch.randn(2, n, 2)
        x_exp = expand_1_to_3(x, mode='linear')
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if ((i == j) and (i == k)):
                        assert(torch.norm(x_exp[:, i, j, k, :] - x[:, i]) < 1e-4)
                    else:
                        assert(torch.norm(x_exp[:, i, j, k, :]) < 1e-4)
