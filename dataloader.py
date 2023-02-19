from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected
from torch_geometric.typing import SparseTensor

import torch
from torch.utils.data import DataLoader
import numpy as np
np.random.seed(0)

import torch_cluster  # noqa
random_walk = torch.ops.torch_cluster.random_walk

OUT_OF_SAMPLE_SIZE = 100
BATCH_SIZE = 128


# Gets full dataset graph with all nodes and edges (for final out-of-sample testing)
def get_full_graph():
    dataset = NodePropPredDataset(name="ogbn-arxiv", root='dataset/')
    return dataset[0]


def get_dataloader(graph, validation=False):
    row, col = graph.edge_index
    n = graph.num_nodes
    edge_adj_matrix = SparseTensor(row=row, col=col, sparse_sizes=(n, n))
    if validation:
        sample_func = lambda x: sample(x, edge_adj_matrix, graph.x, 1, 2)
    else:
        sample_func = lambda x: sample(x, edge_adj_matrix, graph.x)
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


# Returns two tuples of data:
#  - in-sample data: train, validation, and test Data graph objects for in-sample training
#  - out-of-sample data: node indices and features for out of sample testing
def get_datasets():
    dataset = NodePropPredDataset(name="ogbn-arxiv", root='dataset/')
    graph, label = dataset[0]
    node_feat, edge_idx = torch.from_numpy(graph["node_feat"]), torch.from_numpy(graph["edge_index"])

    # Node index splits
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    # np.random.shuffle(test_idx)
    # in_sample_test_idx, out_of_sample_test_idx = torch.from_numpy(test_idx[OUT_OF_SAMPLE_SIZE:]), torch.from_numpy(test_idx[:OUT_OF_SAMPLE_SIZE])

    # In-sample training/validation/testing datasets
    # Note: confirmed that relabel nodes retains node feature order
    train_edges = subgraph(torch.from_numpy(train_idx), edge_idx, relabel_nodes=True)[0]
    train_graph = Data(x=node_feat[train_idx], edge_index=to_undirected(train_edges))
    # train_and_val_idx = np.sort(np.concatenate((train_idx, valid_idx)))
    val_edges = subgraph(torch.from_numpy(val_idx), edge_idx, relabel_nodes=True)[0]
    val_graph = Data(x=node_feat[val_idx], edge_index=to_undirected(val_edges))
    # train_val_test_idx = np.sort(np.concatenate((train_and_val_idx, in_sample_test_idx)))
    test_edges = subgraph(torch.from_numpy(test_idx), edge_idx, relabel_nodes=True)[0]
    test_graph = Data(x=node_feat[test_idx], edge_index=to_undirected(test_edges))

    return train_graph, val_graph, test_graph


# Deprecated
def clean_dataset(full_node_feat, full_edge_idx, split_idxs, in_sample):
    split_node_feat = full_node_feat[split_idxs]
    edge_elem_mask = torch.isin(full_edge_idx, split_idxs)
    edge_mask = torch.sum(edge_elem_mask, axis=0) > in_sample
    split_edge_idxs = full_edge_idx[:, edge_mask]
    return Data(x=split_node_feat, edge_index=split_edge_idxs)


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



def test_randomwalk(graph, val=False):
    edges = []
    for i in range(499):
        edges.append([i, i+1])
    for i in range(0, 498, 2):
        edges.append([i, i + 2])
    # graph = Data(x=torch.arange(500), edge_index=torch.tensor(edges).T)
    dataloader = get_dataloader(graph, val)
    sample_data = []
    for _ in range(10):
        data = next(iter(dataloader))
        sample_data.append(data)
    print(sample_data[0][0].shape)
    print(sample_data[0][1].shape)


if __name__ == "__main__":
    train, val, test = get_datasets()
    print("train", train)
    print("train", train["x"])
    print("train", train.edge_index)
    test_relabel_node()
    print("train")
    test_randomwalk(train)
    print("val")
    test_randomwalk(val, True)
