from ogb.nodeproppred import NodePropPredDataset
import torch
import itertools
from torch.utils.data import DataLoader, Dataset
from torch_geometric.utils import subgraph, to_undirected, negative_sampling
import numpy as np

from dataloader import BATCH_SIZE, clean_edges
from torch_geometric.data import Data


class EdgeDataset(Dataset):
    def __init__(self, graph, node_idxs):
        self.graph = graph
        node_feat, edge_idx = graph.x, torch.from_numpy(graph.edge_index)
        train_edges = subgraph(torch.from_numpy(node_idxs), edge_idx, relabel_nodes=True)[0]
        train_graph = Data(x=node_feat[node_idxs], edge_index=to_undirected(train_edges))
        pos_edges = train_graph.edge_index
        neg_edges = negative_sampling(train_edges, num_neg_samples=pos_edges.shape[1])
        pos_edge_feats = (train_graph.x[pos_edges[0]] - train_graph.x[pos_edges[1]]) ** 2
        neg_edge_feats = (train_graph.x[neg_edges[0]] - train_graph.x[neg_edges[1]]) ** 2
        pos_num, neg_num = pos_edge_feats.shape[0], neg_edge_feats.shape[0]
        dim = pos_edge_feats.shape[1]
        pos_feat_and_label = torch.concat((pos_edge_feats, torch.ones((pos_num, 1))), dim=1)
        neg_feat_and_label = torch.concat((neg_edge_feats, torch.zeros((neg_num, 1))), dim=1)
        self.edge_feat_and_label = torch.stack((pos_feat_and_label, neg_feat_and_label), dim=1).view(pos_num + neg_num, dim+1).detach()
        # self.all_edges_iterator = itertools.combinations(node_idxs, 2)
        # self.num_nodes = node_idxs.shape[0]


    def __len__(self):
        return self.edge_feat_and_label.shape[0]

    def __getitem__(self, idx):
        # edge = next(self.all_edges_iterator)
        # edge_feat = (self.graph.x[edge[0]] - self.graph.x[edge[1]]) ** 2
        # if edge in self.edges:
        #     return {"x" : edge_feat, "y": 1}
        # return {"x" : edge_feat, "y": 0}
        data = self.edge_feat_and_label[idx]
        return {"x" : data[:-1], "y": data[-1]}


class TestEdgeDataset(Dataset):
    def __init__(self, graph, out_sample_idxs):
        # self.graph = graph
        # self.all_edges_iterator = itertools.product(range(graph.num_nodes), out_sample_idxs)
        # self.length = graph.num_nodes * out_sample_idxs.shape[0]
        # self.edges = graph.edge_index.T.tolist()
        node_feat, edge_idx = graph.x, torch.from_numpy(graph.edge_index)
        train_edges = subgraph(torch.from_numpy(node_idxs), edge_idx, relabel_nodes=True)[0]
        train_graph = Data(x=node_feat[node_idxs], edge_index=to_undirected(train_edges))
        pos_edges = train_graph.edge_index
        neg_edges = negative_sampling(train_edges, num_neg_samples=pos_edges.shape[1])
        pos_edge_feats = (train_graph.x[pos_edges[0]] - train_graph.x[pos_edges[1]]) ** 2
        neg_edge_feats = (train_graph.x[neg_edges[0]] - train_graph.x[neg_edges[1]]) ** 2
        pos_num, neg_num = pos_edge_feats.shape[0], neg_edge_feats.shape[0]
        dim = pos_edge_feats.shape[1]
        pos_feat_and_label = torch.concat((pos_edge_feats, torch.ones((pos_num, 1))), dim=1)
        neg_feat_and_label = torch.concat((neg_edge_feats, torch.zeros((neg_num, 1))), dim=1)
        self.edge_feat_and_label = torch.stack((pos_feat_and_label, neg_feat_and_label), dim=1).view(pos_num + neg_num,
                                                                                                     dim + 1).detach()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        edge = next(self.all_edges_iterator)
        edge_feat = (self.graph.x[edge[0]] - self.graph.x[edge[1]]) ** 2
        if edge in self.edges:
            return {"x": edge_feat, "y": 1}
        return {"x": edge_feat, "y": 0}


def get_edge_dataloader(graph):
    dataset = NodePropPredDataset(name="ogbn-arxiv", root='dataset/')
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    in_sample_idxs = np.concatenate((train_idx, val_idx))

    in_sample_dataset = EdgeDataset(graph, in_sample_idxs)
    out_sample_dataset = TestEdgeDataset(graph, test_idx)
    return DataLoader(in_sample_dataset, batch_size=BATCH_SIZE), DataLoader(out_sample_dataset, batch_size=BATCH_SIZE)


if __name__ == '__main__':
    dataset = NodePropPredDataset(name="ogbn-arxiv", root='dataset/')
    graph, label = dataset[0]
    graph = Data(x=graph["node_feat"], edge_index=graph["edge_index"])
    dataloader1, dataloader2 = get_edge_dataloader(graph)
    for step, (x, y) in enumerate(dataloader1):
        print("train: ", step)
    for step, (x, y) in enumerate(dataloader2):
        print("test: ", step)