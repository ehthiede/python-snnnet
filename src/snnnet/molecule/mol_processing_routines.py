import numpy as np
import networkx as nx
from rdkit import Chem
from snnnet.molecule.make_dataset import _DATASET_INFO


def naive_to_mol(adj_mat, v, dataset_type='zinc', ghost_idx=0):
    original_shape = adj_mat.shape
    # adj_mat = adj_mat > threshold
    # v = np.argmax(v, axis=1)
    dataset_info = _DATASET_INFO[dataset_type]

    atom_list = dataset_info['atom_types']
    bond_types = dataset_info['bond_types']
    print(dataset_info)
    adj_mat, v, active_nodes = remove_ghost_atoms(adj_mat, v, ghost_idx)

    # Exception
    if len(adj_mat) == 0:
        raise RuntimeError

    connected_components = get_connected_components(np.sum(adj_mat, axis=2))[0]
    connected_components = np.array(list(connected_components))
    adj_mat = adj_mat[connected_components][:, connected_components]
    v = v[connected_components]

    N = len(v)
    atom_labels = [atom_list[i-1] for i in v]

    mol = Chem.RWMol()
    mol, atom_labels = append_atoms_to_mol(mol, atom_labels)

    new_smaller_adj_mat = np.zeros(adj_mat.shape)
    for i in range(N):
        if atom_labels[i] is None:
            continue
        for j in range(i):
            if atom_labels[j] is None:
                continue
            for k in range(3):
                bond = adj_mat[i, j, k]
                if bond == 0:
                    continue
                else:
                    try:
                        mol.AddBond(i, j, bond_types[k])
                    except RuntimeError:
                        continue
                    new_smaller_adj_mat[i, j, k] = 1.
    new_smaller_adj_mat += new_smaller_adj_mat.swapaxes(0, 1)
    used_indices = active_nodes[connected_components]
    new_adj_mat = expand_submat(new_smaller_adj_mat, original_shape, used_indices)
    return mol, new_adj_mat


def check_validity(mol):
    try:
        smiles = Chem.MolToSmiles(mol)
        mol2 = Chem.MolFromSmiles(smiles)
        if mol2 is not None:
            if mol2.GetNumAtoms() == 0:
                raise RuntimeError
    except RuntimeError:
        return False, None, None
    if mol2 is None:
        return False, None, smiles
    else:
        return True, mol, smiles


def remove_ghost_atoms(adj_mat, v, ghost_idx):
    active_nodes = np.where(v != ghost_idx)[0]
    active_adj_mat = adj_mat[active_nodes][:, active_nodes]
    active_v = v[active_nodes]
    return active_adj_mat, active_v, active_nodes


def get_connected_components(adj_mat, empty_node=21):
    pred_nx = nx.from_numpy_matrix(adj_mat)
    connected_components = sorted(nx.connected_components(pred_nx), key=len, reverse=True)
    return np.array(list(connected_components))


def append_atoms_to_mol(mol, atom_labels):
    N = len(atom_labels)
    for i in range(N):
        if atom_labels[i] is None:
            continue
        # label_i, charge = atom_labels[i]
        label_i = atom_labels[i]
        a = Chem.Atom(label_i)
        # a.SetFormalCharge(charge)
        mol.AddAtom(a)
    return mol, atom_labels


def expand_submat(submat, original_shape, used_indices):
    new_adj_mat = np.zeros(original_shape)
    temp = new_adj_mat[used_indices]
    temp[:, used_indices] = submat
    new_adj_mat[used_indices] = temp
    return new_adj_mat
