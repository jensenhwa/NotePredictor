import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import dataloader
from attri2vec import Attri2Vec
import wandb

sweep_configuration = {
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters':
    {
        # 'epochs': {'values': [15]},
        'lr': {'values': [1e-3]}, #, 1e-4, 1e-5]},
        'num_layers': {'values': [2]}, #, 3, 4]},
        'hidden_dim': {'values': [128]}, #, 256, 512]},
        'output_dim': {'values': [128]}, #, 256, 512]},
        'walk_length': {'values': [5]}, #, 10, 15]},
        'context_size': {'values': [5]}, #, 10, 15]}
     }
}


def train():
    wandb.init()
    print(wandb.config)
    train, val, test = dataloader.get_datasets()
    val_dataloader = dataloader.get_val_dataloader(val)
    train_dataloader = dataloader.get_dataloader(train, wandb.config.walk_length, wandb.config.context_size)
    wandb_logger = WandbLogger(project='note-predictor')
    wandb_logger.experiment.config["hidden_dim"] = wandb.config.hidden_dim
    wandb_logger.experiment.config["output_dim"] = wandb.config.output_dim
    wandb_logger.experiment.config["num_layers"] = wandb.config.num_layers

    attri2vec = Attri2Vec(train.x.shape[1], wandb.config.hidden_dim, wandb.config.output_dim,
                          wandb.config.num_layers, wandb.config.lr)
    wandb.watch(attri2vec, log="all", log_freq=1)
    trainer = pl.Trainer(max_epochs=50, logger=wandb_logger)
    trainer.fit(attri2vec,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    wandb.finish()


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='note-predictor')
    wandb.agent(sweep_id, function=train, count=4)
