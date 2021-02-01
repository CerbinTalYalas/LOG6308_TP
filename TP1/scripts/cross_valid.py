import pandas as pd
import numpy as np
import scipy as sp
import math
import warnings
from scipy.sparse import csr_matrix


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

# Region[Green] Cross-validation

def train(data_train, users, items):
    model = csr_matrix((data_train['rating'], (data_train['user.id'], data_train['item.id']))).toarray()
    return model


def predict(uid, iid, model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg_u = np.mean(model[uid][model[uid] > 0])
        avg_i = np.mean(model[:, iid][model[:, iid] > 0])

    if np.isnan(avg_u):
        avg_u = 0
    if np.isnan(avg_i):
        avg_i = 0

    return (avg_u + avg_i) / 2


def unit_test(test_index, folds, users, items):
    test_set = folds[test_index]
    empty_set = test_set.copy()
    empty_set['rating'] = 0
    train_set = pd.concat(folds[:test_index] + [empty_set] + folds[test_index + 1:])

    model = train(train_set, users, items)

    err = []
    for _, vote in test_set.iterrows():
        pred = predict(vote["user.id"], vote["item.id"], model)
        err.append((vote["rating"] - pred) ** 2)

    return np.mean(err)


def cross_validation(votes, users, items, n_folds=10, verbose=False):
    folds = k_fold(n_folds, votes)
    err = []
    for i in range(n_folds):
        err.append(unit_test(i, folds, users, items))
    if (verbose): print(err)
    return np.mean(err)

# EndRegion
