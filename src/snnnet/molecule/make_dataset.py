"""This script creates the molecule datasets from the raw data files.

For efficient loading of the data into our models, we preprocess the data files into an array format
containing the required data for the models. To read the data and load them into models, please refer
to `snnnet.molecule.molgraph_dataset`.

"""

import argparse
import csv
import functools
import json
import multiprocessing
import os
import re
import shutil
import tarfile

import numpy as np
from rdkit import Chem
import requests
import tqdm


_BOND_TO_INDEX = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
}


# Collects static dataset info
# - atom_types refers to the types of heavy atoms present in the dataset
# - length refers to the total number of molecules in the dataset
# - node_feature_cardinality refers to the number of distinct values for atom types, valence, and formal charge respectively
# - edge_feature_cardinality refers to the number of distinct value for bond types
_DATASET_INFO = {
    'qm9': {
        'atom_types': ['C', 'N', 'O', 'F'],
        'length': 133885,
        'node_feature_cardinality': [4, 4, 3],
        'edge_feature_cardinality': [3],
        'bond_types': [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]
    },
    'zinc': {
        'atom_types': ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I'],
        'length': 249455,
        'node_feature_cardinality': [8, 4, 3],
        'edge_feature_cardinality': [3],
        'bond_types': [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]
    }
}


def smiles_to_graph(smiles: str, dataset_name: str):
    """Converts the given smiles to a graph, and extracts node (atom) labels.

    Parameters
    ----------
    smiles
        The smiles string representing the molecule to process
    dataset_name
        The name of the dataset from which the molecule comes from.

    Returns
    -------
    nodes : np.ndarray
        A num_nodes-by-3 array representing the node features
    edges : np.ndarray
        A num_edges-by-3 array representing the edges, with the first column representing
        the source atom index, the second column representing the bond type, and the
        third column representing the target atom index.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    # Kekulize it
    Chem.Kekulize(mol)
    # remove stereo information, such as inward and outward edges
    Chem.RemoveStereochemistry(mol)

    num_edges = mol.GetNumBonds()
    edges = np.empty((num_edges, 3), dtype=np.int32)

    for i, bond in enumerate(mol.GetBonds()):
        edges[i, 0] = bond.GetBeginAtomIdx()
        edges[i, 1] = _BOND_TO_INDEX[bond.GetBondType()]
        edges[i, 2] = bond.GetEndAtomIdx()

    atom_types = _DATASET_INFO[dataset_name]['atom_types']
    num_nodes = mol.GetNumAtoms()
    nodes = np.empty((num_nodes, 3), dtype=np.int32)

    for i, atom in enumerate(mol.GetAtoms()):
        try:
            nodes[i, 0] = atom_types.index(atom.GetSymbol())
        except ValueError:
            return None, None
        nodes[i, 1] = atom.GetTotalValence()
        nodes[i, 2] = atom.GetFormalCharge()

    return nodes, edges


def extract_smiles_from_qm9_archive(path):
    """Extracts all smiles strings from the contained archive.

    Parameters
    ----------
    path
        A string to the tar file containing the files.

    Yields
    ------
    smiles : bytes
        The smiles string in utf-8 encoding
    filename : str
        The name of the file from which the smiles string was extracted
    """
    archive = tarfile.open(path, 'r')

    # qm9 file names are of the form: 'dsgdb9nsd_000001.xyz'
    # We match the digits (including leading zeros)
    name_index_regex = re.compile(r'^\w+_(\d+)\.xyz$')

    while True:
        f = archive.next()
        if f is None:
            break

        if not f.isfile():
            continue

        lines = archive.extractfile(f).readlines()
        smiles = lines[-2].split(b'\t')[0]
        index = name_index_regex.match(f.name).groups()[0]
        yield smiles, index


def extract_smiles_from_zinc_archive(path):
    """Extracts all smiles from the CSV file.

    Parameters
    ----------
    path
        A string to the csv file containing the files.

    Yields
    ------
    smiles : str
        The smiles string
    index : int
        The index of the record
    """
    with open(path, 'rt') as fd:
        for i, data_item in enumerate(csv.DictReader(fd)):
            smiles = data_item['smiles'].strip()
            yield smiles, i


def _download_file(url, fd):
    """Downloads data from the given url to the given file object.

    Parameters
    ----------
    url : str
        A url string from which to download the data.
    fd : fileobj
        A file object to which to write the data.
    """
    r = requests.get(url, stream=True, allow_redirects=True)
    total = int(r.headers.get('Content-Length', 0))
    r.raw.read = functools.partial(r.raw.read, decode_content=True)

    desc = '(Unknown file size)' if total == 0 else ''

    with tqdm.tqdm.wrapattr(r.raw, "read", total=total, desc=desc) as r_raw:
        shutil.copyfileobj(r_raw, fd)


def _download_file_cached(url, filename, cache_path=None):
    if cache_path is None:
        cache_path = 'data/cache'

    file_path = os.path.join(cache_path, filename)

    if os.path.exists(file_path):
        print('Cached dataset file found at {}, skipping download.'.format(file_path))
        return file_path

    os.makedirs(cache_path, exist_ok=True)

    with open(file_path, 'wb') as fd:
        print('Downloading file to path {}'.format(file_path))
        _download_file(url, fd)

    return file_path


def download_qm9(cache_path=None):
    """Downloads the QM9 dataset.

    This function will skip the download if a cached version is found on the filesystem.

    Parameters
    ----------
    cache_path : str
        Location where to place the qm9 dataset.

    Returns
    -------
    str
        The location of the downloaded dataset.
    """
    return _download_file_cached('https://ndownloader.figshare.com/files/3195389', 'qm9.tar.bz2', cache_path)


def download_qm9_validation_indices(cache_path=None):
    """Downloads the validation indices for QM9."""
    return _download_file_cached(
        'https://raw.githubusercontent.com/microsoft/constrained-graph-variational-autoencoder/master/data/valid_idx_qm9.json',
        'valid_idx_qm9.json', cache_path)


def download_zinc(cache_path=None):
    return _download_file_cached(
        'https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv',
        '250k_rndm_zinc_drugs_clean_3.csv', cache_path)


def download_zinc_validation_indices(cache_path=None):
    return _download_file_cached(
        'https://raw.githubusercontent.com/microsoft/constrained-graph-variational-autoencoder/master/data/valid_idx_zinc.json',
        'valid_idx_zinc.json', cache_path)


def _process_smiles(smiles_and_name, dataset_name):
    smiles, name = smiles_and_name
    nodes, edges = smiles_to_graph(smiles, dataset_name)

    if isinstance(smiles, str):
        smiles = smiles.encode('utf8')

    smiles_buffer = np.frombuffer(smiles, dtype=np.uint8)

    return nodes, edges, smiles_buffer, name


def _merge_arrays(arr_list):
    lengths = np.array([len(arr) for arr in arr_list], dtype=np.int64)
    offsets = np.zeros(lengths.shape[0] + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    values = np.concatenate(arr_list)

    return values, offsets


def _merge_all_arrays(data):
    merged_data = {}

    for k, v in data.items():
        values, offsets = _merge_arrays(v)
        merged_data[k + '_values'] = values
        merged_data[k + '_offsets'] = offsets

    return merged_data


def _process_dataset(archive_path, valid_set_indices, extract_smiles, process_smiles, dataset_info, num_processes: int=None, config: dict=None):
    """Processes a given dataset.

    Parameters
    ----------
    archive_path : str, optional
        Path to the qm9 archive if provided. If not provided, the archive will be downloaded.
    valid_set_path : str, optional
        Path to json file containing validation split. If not provided, validation set indices
        will be downloaded.
    num_processes
        Number of processes to use for multiprocessing. If not provided, uses all cores on the
        current machine.
    config
        Configuration for processing, currently only supports one key: `max_size`, which denotes
        the maximum size of molecules (in atoms) to be included in the dataset.

    Returns
    -------
    dict
        A dictionary containing the processed data.
    """
    if config is None:
        config = dict()
    
    all_smiles = extract_smiles(archive_path)
    pool = multiprocessing.Pool(num_processes)

    graphs = pool.imap_unordered(process_smiles, all_smiles, chunksize=1024)

    splits = {
        'train': {
            'nodes': [],
            'edges': [],
            'smiles': []
        },
        'valid': {
            'nodes': [],
            'edges': [],
            'smiles': []
        }
    }

    num_skipped = 0
    num_processed = 0
    max_size = config.get('max_size', 2 ** 32 - 1)

    for nodes, edges, smiles, index in tqdm.tqdm(graphs, total=dataset_info['length']):
        num_processed += 1
        split_name = 'valid' if index in valid_set_indices else 'train'
        split = splits[split_name]

        if nodes is None or len(nodes) > max_size:
            num_skipped += 1
            continue

        split['nodes'].append(nodes)
        split['edges'].append(edges)
        split['smiles'].append(smiles)
    
    train_data = _merge_all_arrays(splits['train'])
    valid_data = _merge_all_arrays(splits['valid'])

    for data in (train_data, valid_data):
        data['node_feature_cardinality'] = np.array(dataset_info['node_feature_cardinality'], dtype=np.int32)
        data['edge_feature_cardinality'] = np.array(dataset_info['edge_feature_cardinality'], dtype=np.int32)

    print('Done processing dataset. Skipped {} / {} molecules'.format(num_skipped, num_processed))

    return {
        'train': train_data,
        'valid': valid_data
    }


def process_qm9(archive_path=None, valid_set_path=None, num_processes: int=None, cache_path=None, config=None):
    """Processes the qm9 dataset.

    Parameters
    ----------
    archive_path : str, optional
        Path to the qm9 archive if provided. If not provided, the archive will be downloaded.
    valid_set_path : str, optional
        Path to json file containing validation split. If not provided, validation set indices
        will be downloaded.
    num_processes
        Number of processes to use for multiprocessing. If not provided, uses all cores on the
        current machine.
    cache_path : str
        Optional directory where to place (and look for) downloaded archives.
    config : dict, optional
        Optional dictionary containing additional configuration options.

    Returns
    -------
    dict
        A dictionary containing the processed data.
    """
    if archive_path is None:
        archive_path = download_qm9(cache_path)

    if valid_set_path is None:
        valid_set_path = download_qm9_validation_indices(cache_path)

    with open(valid_set_path, 'rt') as fd:
        valid_set_indices = set(json.load(fd)['valid_idxs'])

    return _process_dataset(
        archive_path, valid_set_indices, extract_smiles_from_qm9_archive,
        functools.partial(_process_smiles, dataset_name='qm9'),
        _DATASET_INFO['qm9'], num_processes, config)


def process_zinc(archive_path=None, valid_set_path=None, num_processes: int=None, cache_path=None, config=None):
    """Processes the zinc dataset.

    Parameters
    ----------
    archive_path : str, optional
        Path to the zinc archive if provided. If not provided, the archive will be downloaded.
    valid_set_path : str, optional
        Path to json file containing validation split. If not provided, validation set indices
        will be downloaded.
    num_processes : int
        Number of processes to use for multiprocessing.
    cache_path : str
        Optional directory where to place (and look for) downloaded archives.
    config : dict, optional
        Optional dictionary containing additional configuration options.

    Returns
    -------
    dict
        A dictionary containing the processed data.
    """
    if archive_path is None:
        archive_path = download_zinc(cache_path)

    if valid_set_path is None:
        valid_set_path = download_zinc_validation_indices(cache_path)

    with open(valid_set_path, 'rt') as fd:
        valid_set_indices = set(json.load(fd))

    return _process_dataset(
        archive_path, valid_set_indices, extract_smiles_from_zinc_archive,
        functools.partial(_process_smiles, dataset_name='zinc'),
        _DATASET_INFO['zinc'], num_processes, config)


def _save_splits(splits, path):
    for split_name, split_data in splits.items():
        split_path = path.format(split_name)
        split_path = os.path.abspath(split_path)

        print('Saving split {} at location {}'.format(split_name, split_path))
        np.savez_compressed(split_path, **split_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['qm9', 'zinc'])
    parser.add_argument('--num_processes', type=int, default=None, help='Number of processes to use for multiprocessing.')
    parser.add_argument('--cache_path', type=str, help='Directory in which to place downloaded data files.')
    parser.add_argument('--output_path', type=str, help='Path to use for saving the output. It should contain a single {} pair, which will be replaced by the split name.')
    parser.add_argument('--archive_path', type=str)
    parser.add_argument('--validation_index_path', type=str)
    parser.add_argument('--max_size', type=int, default=None)

    args = parser.parse_args()
    config = {
        'max_size': args.max_size
    }

    if args.dataset == 'qm9':
        splits = process_qm9(args.archive_path, args.validation_index_path, args.num_processes, args.cache_path, config)
    elif args.dataset == 'zinc':
        splits = process_zinc(args.archive_path, args.validation_index_path, args.num_processes, args.cache_path, config)
    else:
        raise ValueError('Unknown dataset')

    output_path = args.output_path

    if output_path is None:
        output_path = os.path.join('data', '{}_{{}}.npz'.format(args.dataset))

    _save_splits(splits, output_path)


if __name__ == '__main__':
    main()
