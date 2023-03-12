import pytorch_lightning as pl
import torch.cuda
from pytorch_lightning.loggers import WandbLogger

import dataloader
from attri2vec import Attri2Vec
from gnn import GNN
from linkprediction import create_embedded_graph, EdgeLogisticRegression
from edge_dataloader import get_edge_dataloader
import wandb
import traceback
import sys

sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters':
    {
        'lr': {'values': [1e-3, 1e-4, 1e-5]},
        'num_layers': {'values': [2, 3, 4]},
        'hidden_dim': {'values': [128, 256, 512]},
        'output_dim': {'values': [128, 256, 512]},
        'walk_length': {'values': [10, 15]},
        'context_size': {'values': [5, 10]},
        'model_type': {'values': ['gnn']}
        # Good values for attri2vec:
#         'lr': {'values': [1e-4]},
#         'num_layers': {'values': [1, 2, 8]},
#         'hidden_dim': {'values': [128, 256]},
#         'output_dim': {'values': [128, 512]},
#         'walk_length': {'values': [5, 10]},
#         'context_size': {'values': [5, 10, 15]}
     }
}


def train():
    wandb.init()
    print(wandb.config)

    try:
        train, val, graph = dataloader.get_datasets()
        val_dataloader = dataloader.get_val_dataloader(val, wandb.config.model_type)
        train_dataloader = dataloader.get_dataloader(train,
                                                     wandb.config.walk_length,
                                                     wandb.config.context_size,
                                                     wandb.config.model_type)
        wandb_logger = WandbLogger(project='note-predictor')
        wandb_logger.experiment.config["hidden_dim"] = wandb.config.hidden_dim
        wandb_logger.experiment.config["output_dim"] = wandb.config.output_dim
        wandb_logger.experiment.config["num_layers"] = wandb.config.num_layers

        # Learn node embeddings
        if wandb.config.model_type == 'attri2vec':
            node_model = Attri2Vec(train.x.shape[1], wandb.config.hidden_dim, wandb.config.output_dim,
                              wandb.config.num_layers, wandb.config.lr)
        elif wandb.config.model_type == 'gnn':
            node_model = GNN(train.x.shape[1], wandb.config.hidden_dim, wandb.config.output_dim,
                              wandb.config.num_layers, wandb.config.lr, train, val)
        else:
            raise ValueError('invalid model_type in wandb config')
        wandb.watch(node_model, log="all", log_freq=1)
        if torch.cuda.is_available():
            trainer = pl.Trainer(max_epochs=50, accelerator="gpu", logger=wandb_logger)
        else:
            trainer = pl.Trainer(max_epochs=6, logger=wandb_logger)

        trainer.fit(node_model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader)

        if hasattr(node_model, 'model'):
            # Link prediction on learned node embeddings
            embedded_test_graph = create_embedded_graph(graph, node_model.model)
            edge_model = EdgeLogisticRegression(wandb.config.output_dim, wandb.config.lr)
            in_sample_edge_dataloader, out_sample_edge_dataloader = get_edge_dataloader(embedded_test_graph)
            if torch.cuda.is_available():
                trainer = pl.Trainer(max_epochs=5, accelerator="gpu", logger=wandb_logger)
            else:
                trainer = pl.Trainer(max_epochs=10, logger=wandb_logger)
            trainer.fit(edge_model, train_dataloaders=in_sample_edge_dataloader, val_dataloaders=out_sample_edge_dataloader)

        wandb.finish()

    except Exception as e:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='note-predictor')
    wandb.agent(sweep_id, function=train)
