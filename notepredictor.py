import dataloader
from attri2vec import Attri2Vec
import pytorch_lightning as pl

if __name__ == "__main__":
    is_data, oos_data = dataloader.get_datasets()
    train, val, test = is_data

    input_dim = is_data.x.shape[1]
    hidden_dim = 128
    output_dim = 128
    num_layers = 2
    walk_length = 10
    context_size = 5

    attri2vec = Attri2Vec(input_dim, hidden_dim, output_dim, num_layers, walk_length, context_size)
    trainer = pl.Trainer(max_epochs=50)
    # TODO: convert is_data to proper format
    trainer.fit(attri2vec, is_data)
