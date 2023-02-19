from unittest import TestCase

import torch

from attri2vec import Attri2Vec


class TestAttri2Vec(TestCase):
    def setUp(self):
        self.num_walks = 15
        self.walk_len = 10
        self.node_emb_len = 128

        self.attri2vec = Attri2Vec(self.node_emb_len, self.node_emb_len, self.node_emb_len, 2, self.walk_len, 5, self.num_walks)

    def test_training_step(self):
        pos_rw = torch.rand((self.num_walks, self.walk_len, self.node_emb_len))
        neg_rw = torch.rand((self.num_walks, self.walk_len, self.node_emb_len))
        print(self.attri2vec.training_step((pos_rw, neg_rw)))

    def test_validation_step(self):
        pos_rw = torch.rand((self.num_walks, self.walk_len, self.node_emb_len))
        neg_rw = torch.rand((self.num_walks, self.walk_len, self.node_emb_len))
        print(self.attri2vec.validation_step((pos_rw, neg_rw)))
