import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from pytorch_lightning import LightningModule
from torchmetrics import Accuracy


# Reference: https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/attri2vec-link-prediction.html

# def in_sample_split(train_dataloader):
#     year_thresh = 2006  # the threshold year for in-sample and out-of-sample set split, which can be 2007, 2008 and 2009
#     subgraph_edgelist = []
#     for ii in range(len(edgelist)):
#         source_index = edgelist["source"][ii]
#         target_index = edgelist["target"][ii]
#         source_year = int(node_data["year"][source_index])
#         target_year = int(node_data["year"][target_index])
#         if source_year < year_thresh and target_year < year_thresh:
#             subgraph_edgelist.append([source_index, target_index])
#     subgraph_edgelist = pd.DataFrame(
#         np.array(subgraph_edgelist), columns=["source", "target"]
#     )
#     subgraph_edgelist["label"] = "cites"  # set the edge type


class Attri2Vec(LightningModule):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            walk_length: int,
            context_size: int,
            walks_per_node: int = 1,
    ):
        super().__init__()
        self.embedding_dim = input_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.EPS = 1e-15

        if num_layers == 1:
            self.model = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        else:
            self.model = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
            for i in range(num_layers - 2):
                self.model.append(nn.Linear(hidden_dim, hidden_dim))
            self.model.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*self.model)

        self.accuracy = Accuracy(task='binary')

    def training_step(self, batch):
        # batch should be the tuple (Tensor(num_walks, rw_len, node_emb_len), Tensor(num_walks, rw_len, node_emb_len))
        # corresponding to the node embeddings of positive and negative random walk samples
        pos_rw, neg_rw = batch

        pos_rw = self.model(pos_rw)
        neg_rw = self.model(neg_rw)

        # Positive loss.
        h_start = pos_rw[:, 0:1, :]
        h_rest = pos_rw[:, 1:, :]  # .contiguous()

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # Negative loss.
        h_start = neg_rw[:, 0:1, :]
        h_rest = neg_rw[:, 1:, :]  # .contiguous()

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()
        self.log("pos_loss", pos_loss, prog_bar=True)
        self.log("neg_loss", neg_loss, prog_bar=True)
        return pos_loss + neg_loss

    def validation_step(self, batch):
        pos_rw, neg_rw = batch

        pos_rw = self.model(pos_rw)

        h_start = pos_rw[:, 0:1, :]
        h_rest = pos_rw[:, 1:, :]

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        assert(len(out) == pos_rw.shape[0] * (pos_rw.shape[1] - 1))
        pos_out = torch.sigmoid(out)

        neg_rw = self.model(neg_rw)
        h_start = neg_rw[:, 0:1, :]
        h_rest = neg_rw[:, 1:, :]  # .contiguous()
        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_out = torch.sigmoid(out)

        labels = torch.cat((torch.ones((len(pos_out))), torch.zeros((len(neg_out)))))

        accuracy = self.accuracy(torch.cat((pos_out, neg_out)), labels)
        self.log("val_acc", accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)
