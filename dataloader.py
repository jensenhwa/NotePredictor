from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

import torch
import numpy as np
np.random.seed(0)

OUT_OF_SAMPLE_SIZE = 100


# Gets full dataset graph with all nodes and edges (for final out-of-sample testing)
def get_full_graph():
    dataset = NodePropPredDataset(name="ogbn-arxiv", root='dataset/')
    return dataset[0]


# Returns two tuples of data:
#  - in-sample data: train, validation, and test Data graph objects for in-sample training
#  - out-of-sample data: node indices and features for out of sample testing
def get_datasets():
    dataset = NodePropPredDataset(name="ogbn-arxiv", root='dataset/')
    graph, label = dataset[0]
    node_feat, edge_idx = torch.from_numpy(graph["node_feat"]), torch.from_numpy(graph["edge_index"])

    # Node index splits
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    np.random.shuffle(test_idx)
    in_sample_test_idx, out_of_sample_test_idx = torch.from_numpy(test_idx[OUT_OF_SAMPLE_SIZE:]), torch.from_numpy(test_idx[:OUT_OF_SAMPLE_SIZE])

    # In-sample training/validation/testing datasets
    # Note: confirmed that relabel nodes retains node feature order
    train_edges = subgraph(torch.from_numpy(train_idx), edge_idx, relabel_nodes=True)[0]
    train_graph = Data(x=node_feat[train_idx], edge_index=train_edges)
    train_and_val_idx = np.sort(np.concatenate((train_idx, valid_idx)))
    val_edges = subgraph(torch.from_numpy(train_and_val_idx), edge_idx, relabel_nodes=True)[0]
    val_graph = Data(x=node_feat[train_and_val_idx], edge_index=val_edges)
    train_val_test_idx = np.sort(np.concatenate((train_and_val_idx, in_sample_test_idx)))
    in_sample_test_edges = subgraph(torch.from_numpy(train_val_test_idx), edge_idx, relabel_nodes=True)[0]
    in_sample_test_graph = Data(x=node_feat[train_val_test_idx], edge_index=in_sample_test_edges)

    return (train_graph, val_graph, in_sample_test_graph), (out_of_sample_test_idx, node_feat[out_of_sample_test_idx])


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


if __name__ == "__main__":
    is_data, oos_data = get_datasets()
    train, val, test = is_data
    print("train", train)
    print("train", train["x"])
    print("train", train.edge_index)
    test_relabel_node()
