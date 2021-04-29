import networkx as nx
import numpy as np


def plot_graph(adj_mat, ax, color=True):
    """
    Plots an N x N adjacency matrix
    """
    pred_nx = nx.from_numpy_matrix(adj_mat)
    if color:
        nx.draw_networkx(pred_nx, ax=ax, node_size=20, with_labels=False, node_color=np.arange(len(adj_mat)), cmap='viridis')
    else:
        nx.draw_networkx(pred_nx, ax=ax, node_size=20, with_labels=False)
    return


def plot_largest_connected(adj_mat, ax):
    """
    Plots the largest connected component in an adjacency matrix.
    """
    pred_nx = nx.from_numpy_matrix(adj_mat)
    connected_components = sorted(nx.connected_components(pred_nx), key=len, reverse=True)
    GC = pred_nx.subgraph(connected_components[0])
    nx.draw_networkx(GC, ax=ax, node_size=100, with_labels=False)
    return
