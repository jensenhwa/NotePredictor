import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import dataloader
from attri2vec import Attri2Vec

if __name__ == "__main__":
    train, val, test = dataloader.get_datasets()
    train_dataloader = dataloader.get_dataloader(train, 20, 10)
    val_dataloader = dataloader.get_val_dataloader(val)

    input_dim = train.x.shape[1]
    hidden_dim = 128
    output_dim = 128
    num_layers = 2
    walk_length = 10
    context_size = 5

    wandb_logger = WandbLogger(project='note-predictor')
    wandb_logger.experiment.config["hidden_dim"] = hidden_dim
    wandb_logger.experiment.config["output_dim"] = output_dim
    wandb_logger.experiment.config["num_layers"] = num_layers

    attri2vec = Attri2Vec(input_dim, hidden_dim, output_dim, num_layers)
    trainer = pl.Trainer(max_epochs=50, logger=wandb_logger)
    trainer.fit(attri2vec,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
