from torch_geometric.data import Data
from sklearn.linear_model import LogisticRegression
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from torch import nn
import torch.nn.functional as F

from dataloader import get_datasets, get_pos_neg_edges
from attri2vec import Attri2Vec


# Single layer logistic regression model
class EdgeLogisticRegression(LightningModule):
    def __init__(self, input_dim: int, lr: float):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.lr = lr

    def training_step(self, batch):
        # batch should be the tuple (edge feature, edge label) where the label is 1 or 0 if the edge is present
        x,y = batch['x'], batch['y']
        x.requires_grad = True
        out = torch.sigmoid(self.linear(x)).squeeze()
        loss = F.binary_cross_entropy(out, y.to(dtype=torch.float32))
        self.log("link_loss", loss, prog_bar=True)
        pred = out > 0.5
        accuracy = torch.count_nonzero(pred == y) / y.shape[0]
        self.log("link_train_acc", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        out = torch.sigmoid(self.linear(x)).squeeze()
        pred = out > 0.5
        accuracy = torch.count_nonzero(pred == y) / y.shape[0]
        self.log("link_val_acc", accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def create_embedded_graph(graph, model):
    orig_feat = torch.from_numpy(graph["node_feat"])
    new_feat = model(orig_feat.unsqueeze(0)).squeeze(0)
    return Data(x=new_feat, edge_index=graph["edge_index"])
