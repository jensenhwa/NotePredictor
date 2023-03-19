import pickle
import sqlite3
from pathlib import Path

# import requests
# import zstandard
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected
from torch_geometric.typing import SparseTensor
import itertools

import torch
from torch.utils.data import DataLoader
import numpy as np
np.random.seed(0)

import torch_cluster  # noqa
random_walk = torch.ops.torch_cluster.random_walk

OUT_OF_SAMPLE_SIZE = 100
BATCH_SIZE = 128


def convert_graph_to_data(graph):
    return Data(x=torch.from_numpy(graph["node_feat"]), edge_index=graph["edge_index"])


# Gets full dataset graph with all nodes and edges (for final out-of-sample testing)
def get_full_graph():
    dataset = NodePropPredDataset(name="ogbn-arxiv", root='dataset/')
    return dataset[0]


def get_val_dataloader(graph, model_type):
    return get_dataloader(graph, 1, 2, model_type)


def get_dataloader(graph, walk_length, context_size, model_type):
    row, col = graph.edge_index
    n = graph.num_nodes
    edge_adj_matrix = SparseTensor(row=row, col=col, sparse_sizes=(n, n))
    if model_type == 'attri2vec':
        sample_func = lambda x: sample(x, edge_adj_matrix, graph.x, walk_length, context_size)
    elif model_type == 'gnn':
        sample_func = lambda x: sample_idxs(x, edge_adj_matrix, graph.x, walk_length, context_size)
    else:
        raise ValueError('invalid model_type')
    return DataLoader(range(edge_adj_matrix.sparse_size(0)),
                          collate_fn=sample_func, batch_size=BATCH_SIZE)


def pos_walk_sample(batch, edge_adj_mat, walk_length, context_size, walks_per_node=10, p=1, q=1):
    batch = batch.repeat(walks_per_node)
    rowptr, col, _ = edge_adj_mat.csr()
    rw = random_walk(rowptr, col, batch, walk_length, p, q)
    if not isinstance(rw, torch.Tensor):
        rw = rw[0]

    walks = []
    num_walks_per_rw = 1 + walk_length + 1 - context_size
    for j in range(num_walks_per_rw):
        walks.append(rw[:, j:j + context_size])
    return torch.cat(walks, dim=0)


def neg_walk_sample(batch, edge_adj_mat, walk_length, context_size, walks_per_node=10, num_negative_samples=1):
    batch = batch.repeat(walks_per_node * num_negative_samples)

    rw = torch.randint(edge_adj_mat.sparse_size(0),
                       (batch.size(0), walk_length))
    rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

    walks = []
    num_walks_per_rw = 1 + walk_length + 1 - context_size
    for j in range(num_walks_per_rw):
        walks.append(rw[:, j:j + context_size])
    return torch.cat(walks, dim=0)


def sample(batch, edge_adj_mat, node_feats, walk_length=20, context_size=10):
    batch = torch.tensor(batch)
    pos_walks_idx = pos_walk_sample(batch, edge_adj_mat, walk_length, context_size)
    neg_walks_idx = neg_walk_sample(batch, edge_adj_mat, walk_length, context_size)
    d1, d2 = pos_walks_idx.shape
    pos_walks_with_emb = torch.index_select(node_feats, 0, pos_walks_idx.flatten()).reshape((d1, d2, -1))
    neg_walks_with_emb = torch.index_select(node_feats, 0, neg_walks_idx.flatten()).reshape((d1, d2, -1))
    return pos_walks_with_emb, neg_walks_with_emb


def sample_idxs(batch, edge_adj_mat, node_feats, walk_length=20, context_size=10):
    batch = torch.tensor(batch)
    pos_walks_idx = pos_walk_sample(batch, edge_adj_mat, walk_length, context_size)
    neg_walks_idx = neg_walk_sample(batch, edge_adj_mat, walk_length, context_size)
    return pos_walks_idx, neg_walks_idx


# Returns train_dataset, val_dataset, (test_graph, full_graph)
#  - in-sample data: train and validation Data graph objects for in-sample training
#  - test data: all nodes but only train/val edges
def get_datasets():
    dataset = NodePropPredDataset(name="ogbn-arxiv", root='dataset/')
    graph, label = dataset[0]
    node_feat, edge_idx = torch.from_numpy(graph["node_feat"]), torch.from_numpy(graph["edge_index"])

    # Node index splits
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    # np.random.shuffle(test_idx)
    # in_sample_test_idx, out_of_sample_test_idx = torch.from_numpy(test_idx[OUT_OF_SAMPLE_SIZE:]), torch.from_numpy(test_idx[:OUT_OF_SAMPLE_SIZE])

    # In-sample training/validation datasets
    # Note: confirmed that relabel nodes retains node feature order
    train_edges = subgraph(torch.from_numpy(train_idx), edge_idx, relabel_nodes=True)[0]
    train_graph = Data(x=node_feat[train_idx], edge_index=to_undirected(train_edges))
    # train_and_val_idx = np.sort(np.concatenate((train_idx, valid_idx)))
    val_edges = subgraph(torch.from_numpy(val_idx), edge_idx, relabel_nodes=True)[0]
    val_graph = Data(x=node_feat[val_idx], edge_index=to_undirected(val_edges))

    # Creates test graph with all train/val edges and separate test nodes
    train_and_val_idx = torch.from_numpy(np.sort(np.concatenate((train_idx, val_idx))))
    train_and_val_edges = clean_edges(edge_idx, train_and_val_idx, True)
    # train_val_test_idx = np.sort(np.concatenate((train_and_val_idx, test_idx)))
    # test_edges = subgraph(torch.arange(graph["num_nodes"]), train_and_val_edges, relabel_nodes=False)[0]
    test_graph = Data(x=node_feat, edge_index=to_undirected(train_and_val_edges))

    return train_graph, val_graph, graph #(test_graph, graph, test_idx)


def get_title(id) -> str:
    url = "https://zenodo.org/record/6511057/files/Papers.txt.zst?download=1"
    data_filepath = Path("dataset") / "papers.txt"
    db_filepath = Path("dataset") / "papers.db"
    if not db_filepath.exists():
        raise FileNotFoundError("Papers database not found")
        # if not data_filepath.exists():
        #     with requests.get(url, stream=True) as r:
        #         r.raise_for_status()
        #         with open(data_filepath, "wb") as f:
        #             dctx = zstandard.ZstdDecompressor()
        #             dctx.copy_stream(r.raw, f)
    con = sqlite3.connect(db_filepath)
    cur = con.cursor()
    res = cur.execute("SELECT PaperTitle from papers WHERE PaperId = ? LIMIT 1", (id,))
    return res.fetchone()[0]


# Returns all edges where either one (not in_sample) or both (in_sample) nodes are within the split idx
def clean_edges(full_edge_idx, split_idxs, in_sample):
    edge_elem_mask = torch.isin(full_edge_idx, split_idxs)
    edge_mask = torch.sum(edge_elem_mask, axis=0) > in_sample
    split_edge_idxs = full_edge_idx[:, edge_mask]
    return split_edge_idxs


# Returns a tuple of edges and edge labels where 1 is existing and 0 is not in graph
# TODO: do we want to shuffle?
def get_pos_neg_edges(graph, in_sample):
    dataset = NodePropPredDataset(name="ogbn-arxiv", root='dataset/')
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    if in_sample:
        idxs = torch.from_numpy(np.concatenate((train_idx, val_idx)))
        all_edges = torch.combinations(idxs)
    else:
        idxs = torch.from_numpy(test_idx)
        all_edges = torch.combinations(torch.arange(graph.num_nodes))

    pos_out_sample_edges = clean_edges(graph.edge_index, idxs, in_sample)

    # TODO: check dimensions
    labels = torch.zeros(all_edges.shape[0])
    labels[all_edges == pos_out_sample_edges] = 1
    return all_edges, labels


def test_relabel_node():
    dataset = NodePropPredDataset(name="ogbn-arxiv", root='dataset/')
    graph, label = dataset[0]
    node_feat, edge_idx = torch.from_numpy(graph["node_feat"]), torch.from_numpy(graph["edge_index"])
    split_idx = dataset.get_idx_split()
    train_idx = torch.from_numpy(split_idx["train"])
    train_edges = subgraph(train_idx, edge_idx, relabel_nodes=True)[0]
    train_graph = Data(x=node_feat[train_idx], edge_index=train_edges)
    random_shuffle_idx = train_idx.clone().numpy()
    np.random.shuffle(random_shuffle_idx)
    random_shuffle_idx = torch.from_numpy(random_shuffle_idx)
    train_shuffle_edges = subgraph(random_shuffle_idx, edge_idx, relabel_nodes=True)[0]
    # Node features need to be in order. Subgraph node indices can be shuffled.
    train_shuffle_graph = Data(x=node_feat[train_idx], edge_index=train_shuffle_edges)
    assert(torch.all(train_graph.edge_index == train_shuffle_graph.edge_index).item())
    assert(torch.all(train_graph.x == train_shuffle_graph.x).item())



def test_randomwalk(graph, walk_length, context_size):
    edges = []
    for i in range(499):
        edges.append([i, i+1])
    for i in range(0, 498, 2):
        edges.append([i, i + 2])
    # graph = Data(x=torch.arange(500), edge_index=torch.tensor(edges).T)
    dataloader = get_dataloader(graph, walk_length, context_size)
    sample_data = []
    for _ in range(10):
        data = next(iter(dataloader))
        sample_data.append(data)
    print(sample_data[0][0].shape)
    print(sample_data[0][1].shape)


if __name__ == "__main__":
    train, val, graph = get_datasets()
    print("train", train)
    print("train", train["x"])
    print("train", train.edge_index)
    test_relabel_node()
    print("train")
    test_randomwalk(train, 20, 10)
    print("val")
    test_randomwalk(val, 1, 2)

