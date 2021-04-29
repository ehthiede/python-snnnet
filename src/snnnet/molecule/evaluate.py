"""Script for evaluating trained Snnet VAE models"""
import argparse
import matplotlib.pyplot as plt
import os
import json
import torch
import numpy as np
from snnnet.molecule import train
from snnnet.molecule import training, molecule_vae
from snnnet.molecule.mol_processing_routines import naive_to_mol, check_validity
from snnnet.visualization import plot_largest_connected as plot_graph
from snnnet.utils import LengthSampler
from rdkit import Chem
from rdkit.Chem import Draw


def _set_up_paths(args):
    """
    Converts paths to absolute file paths and initializes working directory
    (if it doesn't already exist).
    """
    args['directory'] = os.path.abspath(args['directory'])
    args['train_dataset'] = os.path.abspath(args['train_dataset'])
    args['valid_dataset'] = os.path.abspath(args['valid_dataset'])
    return args


def evaluate_model(args):
    total_batch_size = args['batch_size']
    worker_batch_size = total_batch_size

    valid_dataset = train._load_dataset(args['valid_dataset'])
    train_dataset = train._load_dataset(args['train_dataset'])
    valid_dataloader = torch.utils.data.DataLoader(
        # valid_dataset, worker_batch_size, shuffle=True, seed=args['seed'])
        valid_dataset, worker_batch_size, shuffle=True)

    model = molecule_vae.make_molecule_vae(
        args['encoder_channels'], args['decoder_channels'],
        args['latent_channels'], valid_dataset.node_feature_cardinality,
        latent_transform_depth=args['latent_transform_depth'],
        node_embedding_dimension=args['node_embedding_dimension'],
        positional_embedding_dimension=args['positional_embedding_dimension'],
        batch_norm=args['batch_norm'],
        expand_type=args['expand_type'],
        architecture=args['architecture'],
        max_size=train_dataset.pad_size)
    print('made model')

    device = torch.device('cuda')
    state_dict = torch.load(args['saved_state_path'])['harness']['model']
    model.load_state_dict(state_dict)
    model = model.to(device)

    # base_graphs, reconstructed_graphs = create_reconstructions(model, valid_dataloader, num_reconstructions=20)
    # print('finished reconstructions')

    # plot_graph_reconstructions(base_graphs, reconstructed_graphs, args['save_dir'], num_to_plot=20, plot_size=4)

    mu_shape = (train_dataset[0]['edge_features'].shape[0], args['latent_channels'])
    positional_features = (train_dataset[0]['positional_features'])

    positional_generator = LengthSampler(train_dataset, batch_size=worker_batch_size)
    print('MU SHAPE:', mu_shape)
    print('pos feats', positional_features)
    print(train_dataset[1]['positional_features'])
    print(train_dataset[2]['positional_features'])

    generated_graphs = generate_new_graphs(model, mu_shape, positional_generator,
                                           num_generate=5000, batch_size=worker_batch_size)
    molled_graphs, __ = molify_graphs(generated_graphs[0], generated_graphs[1], args['dataset_type'])

    valid_smiles, unique_smiles, long_smiles = evaluate_mols(molled_graphs)

    plot_generated_mols(unique_smiles, args['save_dir'] + '/sample_unique.svg', num_to_plot=25)
    plot_generated_mols(long_smiles, args['save_dir'] + '/sample_long.svg', num_to_plot=25)


def evaluate_mols(graphs):
    N = len(graphs)
    valid_mols = []
    valid_smiles = []
    long_smiles = []
    for mol in graphs:
        is_valid, pmol, smiles = check_validity(mol)
        if is_valid:
            valid_mols.append(pmol)
            valid_smiles.append(smiles)
            if mol.GetNumAtoms() > 10:
                long_smiles.append(smiles)
    unique_smiles = list(set(valid_smiles))
    long_smiles = list(set(long_smiles))
    num_valid = len(valid_smiles)
    num_unique = len(unique_smiles)
    num_long = len(long_smiles)

    print("Number Valid: %d / %d " % (num_valid, N))
    print("Number unique: %d / %d " % (num_unique, N))
    print("Number long: %d / %d " % (num_long, N))
    return valid_smiles, unique_smiles, long_smiles


def molify_graphs(graphs, node_features, dataset_type='zinc'):
    mols = []
    new_adj_mats = []
    for g_i, n_i in zip(graphs, node_features):
        mol, new_adj = naive_to_mol(g_i, n_i, dataset_type=dataset_type)
        mols.append(mol)
        new_adj_mats.append(new_adj)
    return mols, new_adj_mats


def generate_new_graphs(model, mu_shape, positional_generator, num_generate=500, batch_size=32):
    generated_graphs = []
    generated_nfs = []
    with torch.no_grad():
        while batch_size * len(generated_graphs) < num_generate:
            # Generate graph batch
            pred, pred_node = generate_graph_batch_from_noise(model, mu_shape, positional_generator, batch_size)

            # Round to discrete values
            generated_graphs.append((pred > 0).float().detach().cpu())
            generated_nfs.append(torch.argmax(pred_node, dim=-1).detach().cpu())
            print('blah', batch_size * len(generated_graphs), num_generate)
    generated_graphs = torch.cat(generated_graphs, dim=0)[:num_generate].numpy()
    generated_nfs = torch.cat(generated_nfs, dim=0)[:num_generate].numpy()
    return generated_graphs, generated_nfs


def generate_graph_batch_from_noise(model, mu_shape, positional_generator, batch_size):
    # Construct latent representation.
    latent_sample = torch.randn((batch_size,) + mu_shape).cuda()

    # Apply optional positional encoding
    if model.positional_embedding is not None:
        positional_features = positional_generator(batch_size).cuda()
        positional_features = model.positional_embedding(positional_features)
        latent_sample = torch.cat((latent_sample, positional_features), dim=-1)

    # Run Decoder
    latent_sample = latent_sample.to('cuda')
    prediction = model.vae_model.decoder(latent_sample)
    return prediction


def create_reconstructions(model, dataloader, num_reconstructions=5000):
    base_graphs = []
    base_nfs = []
    reconstructed_graphs = []
    reconstructed_nfs = []
    with torch.no_grad():
        for batch in dataloader:
            batch = training._copy_to_device(batch, 'cuda')
            edge_features = batch['edge_features']
            node_features = batch['node_features']
            positional_features = batch['positional_features']
            (logit_edges, logit_nodes), mu, logvar = model(edge_features, node_features, positional_features)
            base_graphs.append(edge_features.cpu())
            base_nfs.append(node_features.cpu())
            discrete_edges = (logit_edges > 0).float()
            reconstructed_graphs.append(discrete_edges.cpu())
            reconstructed_nfs.append(torch.argmax(node_features, dim=-1).cpu())
            # Abort if we have enough molecules
            batch_size = edge_features.shape[0]
            print('batch', len(reconstructed_graphs) * batch_size, num_reconstructions)
            if len(reconstructed_graphs) * batch_size > num_reconstructions:
                print('breaking')
                break
    print('exited')

    base_graphs = torch.cat(base_graphs, dim=0).numpy()
    base_nfs = torch.cat(base_nfs, dim=0).numpy()
    reconstructed_graphs = torch.cat(reconstructed_graphs, dim=0).numpy()
    reconstructed_nfs = torch.cat(reconstructed_nfs, dim=0).numpy()
    return (base_graphs, base_nfs), (reconstructed_graphs, reconstructed_nfs)


def plot_graph_reconstructions(base_graphs, reconstructed_graphs, output_location, num_to_plot=20, plot_size=4):
    # For plotting purposes, we ignore the node features.
    base_graphs = base_graphs[0].sum(axis=-1)
    reconstructed_graphs = reconstructed_graphs[0].sum(axis=-1)
    reconstructed_graphs = (reconstructed_graphs > 0).astype('float')  # account for multiply defined bonds.
    num_plots = num_to_plot // plot_size
    for i in range(num_plots):
        fig, axes = plt.subplots(2, plot_size, figsize=(2 * plot_size, 4))
        for j in range(plot_size):
            idx = i * num_plots + j
            plot_graph(base_graphs[idx], axes[0, j])
            plot_graph(reconstructed_graphs[idx], axes[1, j])
        plt.savefig(output_location + "/adj_mat_reconstruction_%d.png" % i)


def plot_generated_mols(smiles_strings, save_string, num_to_plot=25):
    num_mols = len(smiles_strings)
    choice = np.random.choice(num_mols, num_to_plot).astype('int')
    mol_list = [Chem.MolFromSmiles(smiles_strings[i]) for i in choice]
    print(len(choice), np.max(choice), num_mols, 'choice stats')
    img = Draw.MolsToGridImage(mol_list, molsPerRow=5, subImgSize=(200, 200), useSVG=True)
    # print(type(img))
    # print(img)
    with open(save_string, 'w') as f:
        f.write(img)
    return img


def _parse_args():
    parser = argparse.ArgumentParser(description='Evaluate models')
    parser.add_argument('save_dir', type=str, default='./saved_models/',
                        help='Previous_working_directory')
    parser.add_argument('--train_dataset', default=None, type=str, help='Path to the training dataset')
    parser.add_argument('--valid_dataset', default=None, type=str, help='Path to the validation dataset')
    parser.add_argument('--saved_state_path', default=None, type=str, help='Path to the saved training state')
    parser.add_argument('--num_reconstructions', default=20, type=int, help='Number of reconstructions to plot')
    parser.add_argument('--plot_raw_reconstructions', default=20, type=int, help='Number of reconstructions to plot')
    parser.add_argument('--dataset_type', default='zinc', type=str, help='Dataset to load')
    # parser.add_argument('--num_generations', default=20, type=int, help='Number of molecules to attempt to generate')
    args = parser.parse_args()

    argsdict = {key: val for (key, val) in args.__dict__.items() if val is not None}

    if args.saved_state_path is None:
        argsdict['saved_state_path'] = args.save_dir + "/current_state.pth"

    return argsdict


def main():
    evaluation_args = _parse_args()

    with open(os.path.join(evaluation_args['save_dir'], 'settings.json'), 'r') as f:
        args = json.load(f)

    args.update(evaluation_args)

    evaluate_model(args)


if __name__ == '__main__':
    main()
