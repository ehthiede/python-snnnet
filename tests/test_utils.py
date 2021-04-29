import torch
import pytest
from snnnet import utils


def test_move_from_end():
    a = torch.randn(3, 4, 2, 4, 5, 6)
    a_diag = torch.diagonal(a, dim1=1, dim2=3)
    permuted_a_diag = utils.move_from_end(a_diag, dim=1)
    
    for i in range(4):
        assert(torch.norm(a[:, i, :, i] - permuted_a_diag[:, i]) < 1E-6)


def test_move_to_end():
    a = torch.randn(3, 4, 2, 2)
    a_p = a.permute([0, 2, 3, 1])
    assert(torch.norm(utils.move_to_end(a, 1) - a_p) < 1E-6)

def test_build_idx_3_permutations():
    x_shape = (9,3,3,3,2)
    correct_perms = [(0, 1, 2, 3, 4),
                     (0, 1, 3, 2, 4),
                     (0, 2, 1, 3, 4),
                     (0, 2, 3, 1, 4),
                     (0, 3, 1, 2, 4),
                     (0, 3, 2, 1, 4)]
    
    generated_perms = utils.build_idx_3_permutations(x_shape, 1)
    print(generated_perms)
    for pi in generated_perms:
        assert(pi in correct_perms)
