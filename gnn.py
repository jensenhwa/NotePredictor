import torch
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from torchmetrics import Accuracy


# GNN for in-sample node comparison


class GNN(LightningModule):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            lr: float,
            train: Data,
            val: Data,
    ):
        super().__init__()
        self.lr = lr

        if num_layers == 1:
            self.convs = torch.nn.ModuleList([GCNConv(input_dim, output_dim)])
        else:
            self.convs = torch.nn.ModuleList([GCNConv(input_dim, hidden_dim)])
            for i in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, output_dim))

        self.bns = torch.nn.ModuleList([])
        for i in range(num_layers - 1):
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        self.accuracy = Accuracy(task='binary')

        self.train = train
        self.val = val
        self.train_adj = None
        self.train_feats = None
        self.val_adj = None
        self.val_feats = None
        self.EPS = 0.05

    def prepare_data(self) -> None:
        # https://github.com/Lightning-AI/lightning/issues/13108#issuecomment-1379084281
        device = self.trainer.strategy.root_device
        row, col = self.train.edge_index.to(device)
        n = self.train.num_nodes
        self.train_adj = SparseTensor(row=row, col=col, sparse_sizes=(n, n))
        self.train_feats = self.train.x.to(device)

        row, col = self.val.edge_index.to(device)
        n = self.val.num_nodes
        self.val_adj = SparseTensor(row=row, col=col, sparse_sizes=(n, n))
        self.val_feats = self.val.x.to(device)

    def training_step(self, batch):
        # each batch should be:
        # (positive rw node indices,  negative rw node indices,  node embeddings,    adjacency matrix)
        # (Tensor(num_walks, rw_len), Tensor(num_walks, rw_len), Tensor(num_nodes, node_emb_len), adj)

        # Node embeddings can just correspond to the nodes in the random walks, or the entire training set
        pos_rw, neg_rw = batch

        out = self.train_feats
        for conv, bn in zip(self.convs[:-1], self.bns):
            out = conv(out, self.train_adj)
            out = bn(out)
            out = F.relu(out)
        out = self.convs[-1](out, self.train_adj)

        embedding = nn.Embedding.from_pretrained(out, freeze=False)

        # Positive loss.
        h_start = embedding(pos_rw[:, :1])
        h_rest = embedding(pos_rw[:, 1:])

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # Negative loss.
        h_start = embedding(neg_rw[:, :1])
        h_rest = embedding(neg_rw[:, 1:])  # .contiguous()

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()
        self.log("pos_loss", pos_loss, prog_bar=True)
        self.log("neg_loss", neg_loss, prog_bar=True)
        return pos_loss + neg_loss


    def validation_step(self, batch, batch_idx):
        pos_rw, neg_rw = batch
        out = self.val_feats
        for conv, bn in zip(self.convs[:-1], self.bns):
            out = conv(out, self.val_adj)
            out = bn(out)
            out = F.relu(out)
        out = self.convs[-1](out, self.val_adj)

        embedding = nn.Embedding.from_pretrained(out, freeze=True)

        h_start = embedding(pos_rw[:, :1])
        h_rest = embedding(pos_rw[:, 1:])

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        assert (len(out) == pos_rw.shape[0] * (pos_rw.shape[1] - 1))
        pos_out = torch.sigmoid(out)

        h_start = embedding(neg_rw[:, :1])
        h_rest = embedding(neg_rw[:, 1:])
        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_out = torch.sigmoid(out)

        labels = torch.cat((torch.ones((len(pos_out)), dtype=torch.int, device=pos_out.device),
                            torch.zeros((len(neg_out)), dtype=torch.int, device=pos_out.device)))

        accuracy = self.accuracy(torch.cat((pos_out, neg_out)), labels)
        self.log("val_acc", accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
