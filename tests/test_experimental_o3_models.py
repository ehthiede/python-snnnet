import pytest
import torch
from snnnet.experimental_o3_models import SnResidualEncoder, SnResidualDecoder


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

    assert(torch.allclose(x_out[:, pidx], x_perm_out, atol=1e-6, rtol=1e-4))


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
    
    x_out_perm = x_out[:, pidx][:, :, pidx]
    assert(torch.allclose(x_out_perm, x_perm_out, atol=1e-6, rtol=1e-4))
    if num_node_out_channels > 0:
        assert(torch.norm(x_node_out[:, pidx] - x_perm_node_out) < 1e-3)
