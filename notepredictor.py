import networkx as nx
import pandas as pd
import numpy as np
import os
import random

import dataloader
import attri2vec

if __name__ == "__main__":
    is_data, oos_data = dataloader.get_datasets()
    train, val, test = is_data
    attri2vec.train(train, val, test)
