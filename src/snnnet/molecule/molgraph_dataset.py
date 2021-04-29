"""Read and preprocess data saved by `snnnet.molecule.make_dataset`.

This module contains the main classes to provide the data for training the molecule models.

"""

import numpy as np
import torch
import rdkit.Chem as Chem


class MoleculeData:
    """This class encapsulates a set of arrays representing the molecule data into
    a list-like structure.
    """
    def __init__(self, arrays):
        """Initializes a new molecule dataset from the given array dictionary.

        This function is intended to be used by loading arrays from a file saved
        by `snnnet.molecule.make_dataset`.

        Parameters
        ----------
        arrays : dict
            A dictionary of numpy arrays containing the raw data.
        """
        self._node_offsets = torch.from_numpy(arrays['nodes_offsets'])
        self._node_values = torch.from_numpy(arrays['nodes_values'])
        self._edge_offsets = torch.from_numpy(arrays['edges_offsets'])
        self._edge_values = torch.from_numpy(arrays['edges_values'])
        self._smiles_offsets = torch.from_numpy(arrays['smiles_offsets'])
        self._smiles_values = torch.from_numpy(arrays['smiles_values'])
        self.node_feature_cardinality = torch.from_numpy(arrays['node_feature_cardinality'])
        self.edge_feature_cardinality = torch.from_numpy(arrays['edge_feature_cardinality'])

    def share_memory_(self):
        """Moves the storage of the underlying torch tensors to memory.

        This is used to enable efficient sharing of the dataset across dataloader workers.
        See `torch.Tensor.share_memory_` for details.
        """
        self._node_offsets.share_memory_()
        self._node_values.share_memory_()
        self._edge_offsets.share_memory_()
        self._edge_offsets.share_memory_()
        self._smiles_offsets.share_memory_()
        self._smiles_values.share_memory_()

    def __len__(self):
        return len(self._node_offsets) - 1

    def __getitem__(self, idx):
        return {
            'node_features': self._node_values[self._node_offsets[idx]:self._node_offsets[idx + 1]],
            'graph': self._edge_values[self._edge_offsets[idx]:self._edge_offsets[idx + 1]],
            'smiles': self._smiles_values[self._smiles_offsets[idx]:self._smiles_offsets[idx + 1]].numpy().tobytes().decode('utf8')
        }

    def get_max_nodes_in_graph(self) -> int:
        """Gets the number of nodes in the largest in the current dataset."""
        return np.max(np.diff(self._node_offsets))


class MolGraphDataset(torch.utils.data.Dataset):
    """Dataset of small molecular graphs"""

    def __init__(self, mols: MoleculeData, transform=None, pad_size=None):
        """Creates a new pytorch dataset from the given molecule data.

        Parameters
        ----------
        mols : MoleculeData
            The molecule data to encapsulate as pytorch dataset
        transform : function, None
            If not None, this function is applied to the usual output of this dataset, and
            its return value is the output of this dataset.
        pad_size : int, optional
            The size to which to pad the molecules in the dataset. If None, this is set
            to the size of the largest molecule in the dataset.
        """
        self.mols = mols
        self.transform = transform

        if pad_size is None:
            self.pad_size = self.mols.get_max_nodes_in_graph()
        else:
            self.pad_size = pad_size

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mol = self.mols[idx]

        # 0 is reserved as a padding index
        node_features = mol['node_features'][:, 0] + 1
        node_features = torch.nn.functional.pad(node_features, (0, self.pad_size - node_features.shape[0]))

        positional_features = torch.arange(len(mol['node_features']), dtype=torch.int64) + 1
        positional_features = torch.nn.functional.pad(positional_features, (0, self.pad_size - positional_features.shape[0]))

        edge_features = self._graph_to_adj_mat(mol['graph'])

        sample = {
            'node_features': node_features.type(torch.int64),
            'edge_features': edge_features,
            'positional_features': positional_features}

        if self.transform:
            sample = self.transform(sample)

        return sample

    @property
    def node_feature_cardinality(self):
        """The cardinality of the single node feature used.

        We currently use heavy atom type as the sole node feature, and we also account
        for one value corresponding to a padding value.
        """
        return int(self.mols.node_feature_cardinality[0] + 1)

    @property
    def edge_feature_cardinality(self):
        """The cardinality of the single edge feature used.

        This cardinality refers to the number of distinct edges types.
        """
        return int(self.mols.edge_feature_cardinality[0])

    def _graph_to_adj_mat(self, edge_features):
        adj_mat = torch.zeros(self.pad_size, self.pad_size, self.edge_feature_cardinality)
        edge_features = edge_features.type(torch.int64)

        adj_mat[edge_features[:, 0], edge_features[:, 2], edge_features[:, 1]] = 1
        adj_mat[edge_features[:, 2], edge_features[:, 0], edge_features[:, 1]] = 1
        return adj_mat


class RingAugmentor(object):
    """
    Augments molecules
    """

    def __init__(self, max_num_rings=9):
        super().__init__()
        self.max_rings = max_num_rings

    def __call__(self, sample):
        mol = Chem.MolFromSmiles(sample['smiles'])
        edge_features = sample['edge_features']
        node_features = sample['node_features']
        ring_tuples = mol.GetRingInfo().AtomRings()

        new_ef, new_nf = self.augment(edge_features, node_features, ring_tuples)
        sample['edge_features'] = new_ef
        sample['node_features'] = new_nf
        return sample

    def augment(self, adj_mat, node_features, ring_tuple):
        # Get shapes for reference
        N, __, C_adj = adj_mat.shape
        __, C_node = node_features.shape

        filtered_rings = [ring for ring in ring_tuple if (len(ring) in [5, 6, 7])]

        # Pad adjacency matrix and node feature swith augmented nodes.
        new_adj_mat = torch.zeros(N+self.max_rings, N+self.max_rings, C_adj+3)
        new_adj_mat[:N, :N, :C_adj] = adj_mat
        new_nfs = torch.zeros(N+self.max_rings, C_node+3)
        new_nfs[:N, :C_node] = node_features

        # Add rings
        new_adj_mat, new_nfs = self._add_rings(new_adj_mat, new_nfs, filtered_rings,
                                               N, C_adj, C_node)
        return new_adj_mat, new_nfs

    def _add_rings(self, adj_mat, nfs, filtered_rings, N, C_adj, C_node):
        for i, ring in enumerate(filtered_rings):
            if len(ring) == 5:
                adj_mat[N + i, N + i, C_adj] = 1
                adj_mat[N + i, ring, C_adj] = 1
                adj_mat[ring, N+i, C_adj] = 1
                nfs[N+i, C_node] = 1
            elif len(ring) == 6:
                adj_mat[N + i, N + i, C_adj+1] = 1
                adj_mat[N + i, ring, C_adj+1] = 1
                adj_mat[ring, N+i, C_adj+1] = 1
                nfs[N+i, C_node+1] = 1
            elif len(ring) == 7:
                adj_mat[N + i, N + i, C_adj+2] = 1
                adj_mat[N + i, ring, C_adj+2] = 1
                adj_mat[ring, N+i, C_adj+2] = 1
                nfs[N+i, C_node+2] = 1
            else:
                raise RuntimeError("Ring size is not in [5, 6, 7]")
        return adj_mat, nfs
