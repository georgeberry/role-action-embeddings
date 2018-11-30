import torch
from torch import nn
from torch.nn.modules.module import Module
import numpy as np
import networkx as nx
import pandas as pd
from torch import optim
import seaborn as sns
import random
from itertools import permutations, combinations
from matplotlib import pyplot as plt


class MeanAggregator(Module):
    '''
    This class does the sampling and aggregating given a set of features

    We store everything except the set of nodes to aggregate on the class

    We pass the nodes (batch, neg samples, etc) to forward
    '''
    def __init__(
        self,
        features,
        feature_dim,
        emb_dim,
        n_nbr_samples,
        g,
        dropout=0.5,
        depth=1,
        batchnorm=True,
    ):
        super(MeanAggregator, self).__init__()
        self.feature_dim = feature_dim
        self.depth = depth
        if batchnorm:
            self.fc = nn.Sequential(
                nn.Linear(feature_dim, emb_dim),
                nn.BatchNorm1d(emb_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(feature_dim, emb_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
            )
        self.features = features
        self.n_nbr_samples = n_nbr_samples
        self.g = g # can be any dict like: {node: collection(nbrs)}
        self.random_ = np.random.choice
        self.set_ = set

    def forward(
        self,
        node_list,
        randomize_features=False,
    ):
        '''
        features is (unique_node_dim by feature_dim)
        mask is     (node_list by unique_node_dim)
        '''
        samples = [
            list(self.random_(
                list(self.g[node]),
                self.n_nbr_samples,
                replace=False,
            )) + [node]
            if len(self.g[node]) >= self.n_nbr_samples else list(self.g[node]) + [node]
            for node in node_list
        ]
        unique_nodes_list = list(set.union(*(self.set_(x) for x in samples)))
        unique_nodes_dict = {node: idx for idx, node in enumerate(unique_nodes_list)}

        mask = torch.zeros(len(samples), len(unique_nodes_list))

        row_idxs = []
        col_idxs = []
        for node_idx, node_nbrs in enumerate(samples):
            for alter in node_nbrs:
                row_idxs.append(node_idx)
                col_idxs.append(unique_nodes_dict[alter])
        sampled_features = self.fc(self.features(unique_nodes_list))
        mask[row_idxs, col_idxs] = 1
        mask = mask.div(mask.sum(dim=1).unsqueeze(1))
        if randomize_features:
            mask = mask[torch.randperm(mask.size()[0])]
        return mask.mm(sampled_features)


class EncodingLayer(Module):
    '''
    Forward takes a batch and an aggregator
    It runs one iter of the aggregator and then applies the encoding layer to it
    '''
    def __init__(
        self,
        features,
        feature_dim,
        emb_input_dim,
        emb_dim,
        g,
        agg,
        base_model=None,
        dropout=0.5,
        depth=1,
        batchnorm=True,
    ):
        super(EncodingLayer, self).__init__()
        self.features = features
        self.emb_dim = emb_dim
        self.g = g
        self.agg = agg
        self.depth = depth
        if base_model:
            self.base_model = base_model
        self.fc = nn.Sequential(
            nn.Linear(emb_input_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )


    def forward(self, node_list, randomize_features=False):
        emb = self.agg(
            node_list=node_list,
            randomize_features=randomize_features,
        )
        emb = self.fc(
            emb
        )
        return emb


class MeanModel(Module):
    def __init__(
        self,
        emb_dim,
        n_nbr_samples1,
        n_nbr_samples2,
        g,
        features,
        hidden_dim=8,
        dropout=0.5,
    ):
        super(MeanModel, self).__init__()
        feature_dim = features.size()[1]
        self.agg1 = MeanAggregator(
            features=lambda x: features[x],
            feature_dim=feature_dim,
            emb_dim=hidden_dim,
            n_nbr_samples=n_nbr_samples1,
            g=g,
            dropout=dropout,
            batchnorm=True,
        )
        self.enc1 = EncodingLayer(
            features=lambda x: features[x],
            feature_dim=feature_dim,
            emb_input_dim=hidden_dim,
            emb_dim=hidden_dim,
            g=g,
            agg=self.agg1,
            base_model=None,
            depth=2,
            dropout=dropout,
            batchnorm=True,
        )
        self.agg2 = MeanAggregator(
            features=lambda x: self.enc1(x),
            feature_dim=hidden_dim,
            emb_dim=hidden_dim,
            n_nbr_samples=n_nbr_samples2,
            g=g,
            dropout=dropout,
            batchnorm=True,
        )
        self.enc2 = EncodingLayer(
            features=lambda x: self.enc1(x),
            feature_dim=hidden_dim,
            emb_input_dim=hidden_dim,
            emb_dim=emb_dim,
            g=g,
            agg=self.agg2,
            base_model=self.enc1,
            depth=1,
            dropout=dropout,
            batchnorm=True,
        )
        self.model = self.enc2.apply(init_weights)

    def forward(self, node_list, randomize_features=False):
        if self.model.training:
            return self.model(node_list, randomize_features)
        else:
            return torch.cat(
                (
                    self.enc2(node_list, False),
                    self.enc1(node_list, False),
                ),
                dim=1,
            )


def sigmoid_loss(emb_u, emb_v, emb_neg, pos_weight):
    logsigmoid = nn.LogSigmoid()
    emb_v = emb_v.view(emb_u.size()[0], -1, emb_u.size()[1])
    emb_neg = emb_neg.view(emb_u.size()[0], -1, emb_u.size()[1])
    pos = logsigmoid((emb_u.unsqueeze(1) * emb_v).sum(dim=2)).mean()
    neg = logsigmoid(-(emb_u.unsqueeze(1) * emb_neg).sum(dim=2)).mean()
    return - (pos + neg)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)



def pad_features(features, feature_max_len):
    while len(features) < feature_max_len:
        features.append(0)
    if len(features) > feature_max_len:
        features = np.random.choice(
            features,
            size=feature_max_len,
        )
    return features
