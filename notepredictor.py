import networkx as nx
import pandas as pd
import numpy as np
import os
import random

import dataloader
import attri2vec

if __name__ == "__main__":
    train, val, test = dataloader.get_datasets()

    attri2vec.train(train, val, test)
