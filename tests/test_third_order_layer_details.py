"""Tests for implementation details of third-order layers"""

import torch
import pytest

from snnnet import third_order_layers


def test_second_order_contractions():
    gen = torch.Generator()
    gen.manual_seed(42)

    x = torch.randn((8, 4, 4, 4, 6), generator=gen)

    diags_ref, stacked_ref = third_order_layers._compute_second_order_contractions_reference(x, [1, 2, 3])
    diags, stacked = third_order_layers._compute_second_order_contractions(x, [1, 2, 3])

    for d, dr in zip(diags, diags_ref):
        torch.testing.assert_allclose(d, dr)

    torch.testing.assert_allclose(stacked, stacked_ref)


@pytest.mark.parametrize('reduction', ['mean', 'sum', 'sqrt'])
def test_second_order_contractions_backward(reduction):
    gen = torch.Generator()
    gen.manual_seed(42)

    x = torch.randn((2, 3, 3, 3, 2), generator=gen, requires_grad=True, dtype=torch.float64)

    assert torch.autograd.gradcheck(
        lambda x: third_order_layers._SecondOrderContractions.apply(x, reduction), x)


def test_third_order_mix():
    gen = torch.Generator()
    gen.manual_seed(42)

    x = torch.randn((2, 3, 3, 3, 2), generator=gen, requires_grad=True)
    layer = torch.nn.Linear(6 * 2, 4)

    output_ref = third_order_layers._third_order_permutation_matmul_reference(layer, x)
    output = third_order_layers._third_order_permutation_matmul(layer, x)

    torch.testing.assert_allclose(output, output_ref)


def test_expand_mix():
    gen = torch.Generator()
    gen.manual_seed(42)

    mixed_full = torch.randn((4, 3, 3, 3, 2), generator=gen, requires_grad=True)
    mixed_full_ref = mixed_full.clone()
    mixed_slices = [torch.randn((4, 3, 3, 2), generator=gen, requires_grad=True) for _ in range(3)]
    mixed_diag = torch.randn((4, 3, 2), generator=gen, requires_grad=True)

    third_order_layers._expand_mix_reference(mixed_full_ref, mixed_slices, mixed_diag, 1, 2, 3)
    third_order_layers._expand_mix(mixed_full, mixed_slices, mixed_diag, 1, 2, 3)

    torch.testing.assert_allclose(mixed_full, mixed_full_ref)
