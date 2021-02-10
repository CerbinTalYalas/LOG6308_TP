import warnings
from statistics import NormalDist

import numpy as np
import pandas as pd
import math

from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans


def k_fold(nb_fold, data):
    shuffle_data = data.sample(frac=1, random_state=0)
    folds = []
    item_per_fold = math.floor(len(shuffle_data) / nb_fold)
    start, end = 0, item_per_fold
    for i in range(nb_fold):
        folds.append(shuffle_data.iloc[start:end, :])
        start = end
        end = len(data) if (i + 1 == nb_fold - 1) else end + item_per_fold
    return folds


def train_test_spit(test_index, folds):
    test = folds[test_index]
    empty_set = test.copy()
    empty_set['rating'] = 0
    train = pd.concat(folds[:test_index] + [empty_set] + folds[test_index + 1:])

    return train, test


def compute_clusters_mean(data, clusters, nb_cluster):
    result = []
    for i in range(nb_cluster):
        mask = np.ma.masked_where(clusters == i, clusters).mask
        masked_data = data[mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.true_divide(masked_data.sum(0), (masked_data != 0).sum(0))
        result.append(mean)

    return result


def predict(uid, iid, clusters, clusters_mean):
    cluster_num = clusters[uid]
    return clusters_mean[cluster_num][iid]


def mse(test_set, clusters, clusters_mean):
    err = []
    count = 0
    for _, vote in test_set.iterrows():
        pred = predict(vote["user.id"], vote["item.id"], clusters, clusters_mean)
        if np.isnan(pred): count += 1
        err.append((vote["rating"] - pred) ** 2)

    print(count)
    return np.nanmean(err)


def agglomeration(data, n_folds, n_cluster):
    folds = k_fold(n_folds, data)
    err = []
    for i in range(n_folds):
        train_set_pd, test_set_pd = train_test_spit(i, folds)
        train_set_np = csr_matrix((train_set_pd['rating'], (train_set_pd['user.id'], train_set_pd['item.id']))).toarray()
        random_n = NormalDist(0, 1e-4).samples(train_set_np.size, seed=0)
        random_n = np.reshape(random_n, train_set_np.shape)

        train_set_cor = train_set_np + random_n

        cor = np.corrcoef(train_set_cor)
        kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(cor)
        clusters = kmeans.labels_
        clusters_mean = compute_clusters_mean(train_set_np, clusters, n_cluster)
        err.append(mse(test_set_pd, clusters, clusters_mean))

    return np.mean(err)



"""random_N = np.random.rand(*train_set_np.shape)
random_N = random_N - 0.5
random_N = random_N/10000

train_set_np = train_set_np + random_N"""

votes = pd.read_csv("./data/votes.csv", sep="|")
n_votes = len(votes)
votes['user.id'] -= 1
votes['item.id'] -= 1

folds = k_fold(5, votes)
err = []
for i in range(1):
    train_set_pd, test_set_pd = train_test_spit(i, folds)
    train_set_np = csr_matrix((train_set_pd['rating'], (train_set_pd['user.id'], train_set_pd['item.id']))).toarray()
    random_n = NormalDist(0, 1e-4).samples(train_set_np.size, seed=0)
    random_n = np.reshape(random_n, train_set_np.shape)
    """random_n = np.random.rand(*train_set_np.shape)
    random_n = random_n - 0.5
    random_n = random_n / 10000"""

    train_set_cor = train_set_np + random_n

    cor = np.corrcoef(train_set_cor)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(cor)
    clusters = kmeans.labels_
    clusters_mean = compute_clusters_mean(train_set_np, clusters, 5)
    a = mse(test_set_pd, clusters, clusters_mean)
    print(a)
    err.append(a)

res = np.mean(err)