import warnings
from statistics import NormalDist

import numpy as np
import pandas as pd
import math

from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans


# Region[Red] K-fold

def k_fold(nb_fold, data):
    shuffle_data = data.sample(frac=1, random_state=10)
    folds = []
    item_per_fold = math.floor(len(shuffle_data) / nb_fold)
    start, end = 0, item_per_fold
    for i in range(nb_fold):
        folds.append(shuffle_data.iloc[start:end, :])
        start = end
        end = len(data) if (i + 1 == nb_fold - 1) else end + item_per_fold
    return folds


# EndRegion

# Region[Green] Clustering

def train_test_spit(test_index, folds):
    test = folds[test_index]
    empty_set = test.copy()
    empty_set['rating'] = 0
    train = pd.concat(folds[:test_index] + [empty_set] + folds[test_index + 1:])

    return train, test


def mean_without_zero(data, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.true_divide(data.sum(axis), (data != 0).sum(axis))


def compute_clusters_mean(data, clusters, nb_cluster):
    result = []
    for i in range(nb_cluster):
        mask = np.ma.masked_where(clusters == i, clusters).mask
        masked_data = data[mask]
        means = mean_without_zero(masked_data, 0)
        result.append(means)

    return result


def fill_missing(data):
    avg_u = mean_without_zero(data, axis=1)
    avg_i = mean_without_zero(data, axis=0)
    for uid, user in enumerate(data):
        for iid, item in enumerate(user):
            if item == 0:
                val = 0
                div = 0
                if not np.isnan(avg_u[uid]):
                    val += avg_u[uid]
                    div += 1
                if not np.isnan(avg_i[iid]):
                    val += avg_i[iid]
                    div += 1
                data[uid][iid] = val / div


def predict(uid, iid, clusters, clusters_mean):
    cluster_num = clusters[uid]
    return clusters_mean[cluster_num][iid]


def mse(test_set, clusters, clusters_mean):
    err = []
    for _, vote in test_set.iterrows():
        pred = predict(vote["user.id"], vote["item.id"], clusters, clusters_mean)
        dif = (vote["rating"] - pred)
        err.append(dif ** 2)
    return np.nanmean(err)


def agglomeration(data, n_folds, n_cluster):
    folds = k_fold(n_folds, data)
    err = []
    full_matrix = csr_matrix((data['rating'], (data['user.id'], data['item.id']))).toarray()

    missing_val = np.ma.masked_where(full_matrix == 0, full_matrix).mask

    full_matrix = full_matrix.astype(float)
    fill_missing(full_matrix)

    for i in range(n_folds):
        train_set_pd, test_set_pd = train_test_spit(i, folds)
        train_set_np = csr_matrix(
            (train_set_pd['rating'], (train_set_pd['user.id'], train_set_pd['item.id']))).toarray()
        random_n = NormalDist(0, 1e-4).samples(train_set_np.size, seed=0)
        random_n = np.reshape(random_n, train_set_np.shape)

        train_set_cor = train_set_np + random_n
        cor = np.corrcoef(train_set_cor)

        kmeans = KMeans(n_clusters=n_cluster, random_state=5).fit(cor)
        clusters = kmeans.labels_

        train_set_np = train_set_np.astype(float)
        train_set_np[missing_val] = full_matrix[missing_val]

        clusters_mean = np.array(compute_clusters_mean(train_set_np, clusters, n_cluster))
        err.append(mse(test_set_pd, clusters, clusters_mean))

    return np.mean(err)

# EndRegion
