import stellargraph as sg
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import Attri2VecLinkGenerator, Attri2VecNodeGenerator
from stellargraph.layer import Attri2Vec, link_classification

from tensorflow import keras

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import to_networkx
import random


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


def train(train_graph, val_graph, test_graph):
    nx_train = to_networkx(train_graph, node_attrs=['x'])

    G_train = sg.StellarGraph.from_networkx(nx_train, node_features='x')
    print(G_train.info())
