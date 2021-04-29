import pytest
import torch

from snnnet import third_order_vae


@pytest.mark.parametrize('channel_list', [[3, 2], [2, 2, 2]])
@pytest.mark.parametrize('node_channels', [0, 1, 3])
def test_encoder_equivariance(channel_list, node_channels):
    n = 3
    pidx = torch.randperm(n)  # random permutation
    x = torch.randn(4, n, n, 2)
    x_perm = x[:, pidx][:, :, pidx]
    if node_channels > 0:
        x_node = torch.randn(4, n, node_channels)
        x_node_perm = x_node[:, pidx]
    else:
        x_node = x_node_perm = None

    snc = third_order_vae.SnEncoder(2, channel_list, 3, node_channels=node_channels)

    # Permute the output
    x_out = snc(x, x_node=x_node)
    x_out_perm = x_out[:, pidx]

    # Permute the input
    x_perm_out = snc(x_perm, x_node=x_node_perm)
    assert torch.allclose(x_out_perm, x_perm_out, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize('channel_list', [[3, 2], [2]])
@pytest.mark.parametrize('node_channels', [0, 1, 3])
@pytest.mark.parametrize('decoder_expand_type', ["individual", "linear", "direct"])
def test_equivariance(channel_list, node_channels, decoder_expand_type):
    n = 3
    pidx = torch.randperm(n)  # random permutation
    x = torch.randn(4, n, 2)
    x_perm = x[:, pidx]

    snc = third_order_vae.SnDecoder(2, channel_list, 3, node_out_channels=node_channels, expand_type=decoder_expand_type)

    x_out, x_node_out = snc(x)
    x_perm_out, x_perm_node_out = snc(x_perm)

    x_out_perm = x_out[:, pidx][:, :, pidx]
    assert torch.allclose(x_out_perm, x_perm_out, atol=1e-6, rtol=1e-4)

    if node_channels > 0:
        x_node_out_perm = x_node_out[:, pidx]
        assert torch.allclose(x_node_out_perm, x_perm_node_out, atol=1e-6, rtol=1e-4)
