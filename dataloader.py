from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.data import Data

import torch


def clean_dataset(full_node_feat, full_edge_idx, split_idxs, in_sample):
    split_node_feat = full_node_feat[split_idxs]
    edge_elem_mask = torch.isin(full_edge_idx, split_idxs)
    edge_mask = torch.sum(edge_elem_mask, axis=0) > in_sample
    split_edge_idxs = full_edge_idx[:, edge_mask]
    return Data(x=split_node_feat, edge_index=split_edge_idxs)

def get_datasets():
    dataset = NodePropPredDataset(name = "ogbn-arxiv", root = 'dataset/')
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = torch.from_numpy(split_idx["train"]), torch.from_numpy(split_idx["valid"]), torch.from_numpy(split_idx["test"])
    graph, label = dataset[0]
    node_feat, edge_idx = torch.from_numpy(graph["node_feat"]), torch.from_numpy(graph["edge_index"])
    train_data = clean_dataset(node_feat, edge_idx, train_idx, True)
    val_data = clean_dataset(node_feat, edge_idx, valid_idx, False)
    test_data = clean_dataset(node_feat, edge_idx, test_idx, False)
    return train_data, val_data, test_data


if __name__ == "__main__":
    train, val, test = get_datasets()
    print("train", train)
    print("train", train["x"])
    print("train", train.edge_index)
