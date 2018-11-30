import torch
from torch import nn
from torch.nn.modules.module import Module
import numpy as np
import networkx as nx
from collections import defaultdict
import pandas as pd
from torch import optim
import seaborn as sns
import random
from itertools import permutations, combinations
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from matplotlib import pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

def pad_features(features, feature_max_len):
    while len(features) < feature_max_len:
        features.append(0)
    if len(features) > feature_max_len:
        features = np.random.choice(
            features,
            size=feature_max_len,
        )
    return features

def init_weights(m):
    try:
        torch.nn.init.xavier_uniform_(m.weight)
    except:
        pass


def generate_house():
    g = nx.Graph()
    g.add_path(range(15))
    g.add_edge(0,14)
    for x in g.node():
        g.node[x]['label'] = 'ring'
    start = 15
    connector = 0
    # make houses
    for _ in range(5):
        g.add_edges_from(combinations(range(start, start + 4), 2))
        g.add_edges_from([(start + 2, start + 4),(start + 3, start + 4),(start + 3, connector)])

        g.node[start]['label'] = 'base'
        g.node[start+1]['label'] = 'base'
        g.node[start+2]['label'] = 'window'
        g.node[start+3]['label'] = 'door'
        g.node[start+4]['label'] = 'roof'
        g.node[connector]['label'] = 'connector'

        start += 5
        connector += 3

    g = nx.convert_node_labels_to_integers(g)
    n_edges = len(g.edges())
    node_list = list(g.node())
    structural_features = []
    other_features = []
    feature_dim = max(dict(g.degree).values())
    for node in g.node():
        structural_features.append(
            [g.degree(node)] + pad_features(sorted([g.degree(x) for x in g[node]], reverse=True), feature_dim)
        )
    structural_features = torch.FloatTensor(structural_features)
    other_features = torch.FloatTensor(other_features)
    node_labels = [g.node[node]['label'] for node in g.node]
    return g, structural_features


def generate_barbell():
    color = 0
    g = nx.Graph()
    g.add_nodes_from([0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18], label=color)
    color += 1
    g.add_nodes_from([9,19], label=color)
    color += 1
    for idx in range(20, 25):
        g.add_node(idx, label=color)
        color += 1
    for idx in range(25, 30):
        color -= 1
        g.add_node(idx, label=color)
    g.add_edges_from(combinations([0,1,2,3,4,5,6,7,8,9], 2))
    g.add_edges_from(combinations([10,11,12,13,14,15,16,17,18,19], 2))
    g.add_path([9,20,21,22,23,24,25,26,27,28,29,19])
    g = nx.convert_node_labels_to_integers(g)
    n_edges = len(g.edges())
    node_list = list(g.node())
    structural_features = []
    other_features = []
    feature_dim = max(dict(g.degree).values())
    for node in g.node():
        structural_features.append(
            [g.degree(node)] + pad_features(sorted([g.degree(x) for x in g[node]], reverse=True), feature_dim)
        )
    structural_features = torch.FloatTensor(structural_features)
    other_features = torch.FloatTensor(other_features)
    node_labels = [g.node[node]['label'] for node in g.node]
    return g, structural_features
