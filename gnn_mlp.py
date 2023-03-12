import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_sparse import SparseTensor

from gnn import GNN


class MLP(LightningModule):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            lr: float,
            train: Data,
            val: Data,
            gnn: GNN
    ):
        super().__init__()
        self.lr = lr
        if num_layers == 1:
            self.model = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        else:
            self.model = nn.ModuleList([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            for i in range(num_layers - 2):
                self.model.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            self.model.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*self.model)

        self.train = train
        self.val = val
        self.train_adj = None
        self.train_feats = None
        self.val_adj = None
        self.val_feats = None
        self.gnn = gnn
        self.train_y = None
        self.val_y = None

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
        self.train_y = get_gnn_emb(self.gnn, self.train_feats, self.train_adj)
        self.val_y = get_gnn_emb(self.gnn, self.val_feats, self.val_adj)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_feats, batch_size=self.train_feats.shape[0])

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_feats, batch_size=self.val_feats.shape[0])

    def training_step(self, batch, batch_idx):
        out = self.model(self.train_feats)
        loss = F.mse_loss(out, self.train_y)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(self.val_feats)
        loss = F.mse_loss(out, self.val_y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def get_gnn_emb(gnn, out, adj):
    with torch.no_grad():
        for conv, bn in zip(gnn.convs[:-1], gnn.bns):
            out = conv(out, adj)
            out = bn(out)
            out = F.relu(out)
        return gnn.convs[-1](out, adj)


def train_gnn_mlp(gnn, wandb_logger, input_dim: int,
                  hidden_dim: int,
                  output_dim: int,
                  num_layers: int,
                  lr: float,
                  train: Data,
                  val: Data, ):
    model = MLP(input_dim,
                hidden_dim,
                output_dim,
                num_layers,
                lr,
                train,
                val,
                gnn)

    if torch.cuda.is_available():
        trainer = pl.Trainer(max_epochs=50, accelerator="gpu", logger=wandb_logger)
    else:
        trainer = pl.Trainer(max_epochs=6, logger=wandb_logger)
    trainer.fit(model)
    return model
