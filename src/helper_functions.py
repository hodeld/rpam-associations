import math

import numpy as np
from os import path

import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA
import pickle
import random
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

from src.terms_lexica import get_equally_distributed_sst_dataset


# Helper Functions

def std_deviation(J):
    mean_J = np.mean(J)
    var_J = sum([(j - mean_J) ** 2 for j in J])
    return (np.sqrt(var_J / (len(J) - 1)))


def create_permutation(a, b):
    permutation = random.sample(a + b, len(a + b))
    return permutation[:int(len(permutation) * .5)], permutation[int(len(permutation) * .5):]


def flatten(l):
    return [item for sublist in l for item in sublist]


def softmax_stable(x):
    return (np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())


def cohens_d(distribution_x, distribution_y):
    coh_d = (np.mean(distribution_x) - np.mean(distribution_y)) / std_deviation(distribution_x + distribution_y), \
            std_deviation(distribution_x + distribution_y)
    return coh_d


def convert_cont_to_discr_dist(data, ground_truth_humans):  #
    num_bins = 2
    quant1 = ground_truth_humans.value_counts()[0] / ground_truth_humans.value_counts().sum()
    if num_bins == 3:
        quant2 = ground_truth_humans.value_counts()[1] / ground_truth_humans.value_counts().sum()
        quantiles = np.quantile(data, [quant1, quant1 + quant2])
        bins = np.array([np.min(data), *quantiles, np.max(data)])
    else:
        quantiles = np.quantile(data, [quant1])
        bins = np.array([np.min(data), *quantiles, np.max(data)])
    # bins = np.linspace(np.min(data), np.max(data), num_bins + 1)
    labels = np.digitize(data, bins[:num_bins], right=False) - 1
    assert ground_truth_humans.value_counts()[0] == pd.Series(labels).value_counts()[0]
    return pd.Series(labels)


def get_equally_distributed_dataset(df, label_name='ground_truth'):
    nr_i = df[label_name].value_counts().min()
    dfs = []
    for label in [0, 1]:
        dfi = df[df[label_name] == label]
        dfi = dfi.sample(n=nr_i, random_state=1)
        dfs.append(dfi)
    df = pd.concat(dfs, axis=0)
    return df


def f1_balanced_set(valence, ground_truth):
    """take f1 from a balanced dataset in order to compare to a dummy classifier"""

    df = pd.DataFrame(zip(ground_truth, valence), columns=['ground_truth', 'valence'], index=ground_truth.index)
    df = get_equally_distributed_dataset(df, 'ground_truth')
    valence = df['valence']
    ground_truth = df['ground_truth']
    labels = convert_cont_to_discr_dist(valence, ground_truth)
    assert ground_truth.value_counts()[0] == labels.value_counts()[0]
    f1 = f1_score(y_true=ground_truth, y_pred=labels)
    return f1


def f1_balanced_set_from_labels(valence, ground_truth):
    """take f1 from a balanced dataset in order to compare to a dummy classifier"""
    df = pd.DataFrame(zip(ground_truth, valence), columns=['ground_truth', 'valence'], index=ground_truth.index)
    df = get_equally_distributed_dataset(df, 'ground_truth')
    labels = df['valence']
    ground_truth = df['ground_truth']
    f1 = f1_score(y_true=ground_truth, y_pred=labels)
    return f1
