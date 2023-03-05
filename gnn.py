import torch
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.nn import GCNConv
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

    def training_step(self, batch):
        # each batch should be:
        # (positive rw node indices,  negative rw node indices,  node embeddings,    adjacency matrix)
        # (Tensor(num_walks, rw_len), Tensor(num_walks, rw_len), Tensor(num_nodes, node_emb_len), adj)

        # Node embeddings can just correspond to the nodes in the random walks, or the entire training set
        pos_rw, neg_rw, x, adj_t = batch

        out = x
        for conv, bn in zip(self.convs[:-1], self.bns):
            out = conv(out, adj_t)
            out = bn(out)
            out = F.relu(out)
        out = self.convs[-1](out, adj_t)

        embedding = nn.Embedding.from_pretrained(out, freeze=True)

        # Positive loss.
        h_start = embedding(pos_rw[:, 0:1, :])
        h_rest = embedding(pos_rw[:, 1:, :])

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # Negative loss.
        h_start = embedding(neg_rw[:, 0:1, :])
        h_rest = embedding(neg_rw[:, 1:, :])  # .contiguous()

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()
        self.log("pos_loss", pos_loss, prog_bar=True)
        self.log("neg_loss", neg_loss, prog_bar=True)
        return pos_loss + neg_loss


    def validation_step(self, batch, batch_idx):
        pos_rw, neg_rw, x, adj_t = batch
        out = x
        for conv, bn in zip(self.convs[:-1], self.bns):
            out = conv(out, adj_t)
            out = bn(out)
            out = F.relu(out)
        out = self.convs[-1](out, adj_t)

        embedding = nn.Embedding.from_pretrained(out, freeze=True)

        h_start = embedding(pos_rw[:, 0:1, :])
        h_rest = embedding(pos_rw[:, 1:, :])

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        assert (len(out) == pos_rw.shape[0] * (pos_rw.shape[1] - 1))
        pos_out = torch.sigmoid(out)

        h_start = embedding(neg_rw[:, 0:1, :])
        h_rest = embedding(neg_rw[:, 1:, :])
        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_out = torch.sigmoid(out)

        labels = torch.cat((torch.ones((len(pos_out)), dtype=torch.int, device=pos_out.device),
                            torch.zeros((len(neg_out)), dtype=torch.int, device=pos_out.device)))

        accuracy = self.accuracy(torch.cat((pos_out, neg_out)), labels)
        self.log("val_acc", accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
