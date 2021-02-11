import numpy as np
import pandas as pd
import scipy as sp
import math
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

#Region[Magenta] SVD Decomposition

def svd_decomp(model):
    u,s,vh = np.linalg.svd(model, full_matrices = False)
    return u, s, vh

def svd_decomp_reduc(model, k):
    u, s, vh = svd_decomp(model)
    if k > len(s):
        k = len(s)
    s = s * (k*[1.0]+(len(s)-k)*[0.0])
    return u, np.diag(s), vh

#EndRegion

#Region[Green] Cross validation

def train(data_train, k):
    model_raw = csr_matrix((data_train['rating'], (data_train['user.id'], data_train['item.id']))).toarray()
    u, s, vh = svd_decomp_reduc(model_raw, k)
    svd_matrix = u @ s @ vh
    return svd_matrix

def unit_test(test_index, folds, k):
    test_set = folds[test_index]
    empty_set = test_set.copy()
    empty_set['rating'] = 0
    train_set = pd.concat(folds[:test_index] + [empty_set] + folds[test_index + 1:])

    model = train(train_set, k)

    err = []
    for _, vote in test_set.iterrows():
        err.append((vote['rating'] - model[vote['user.id'],vote['item.id']]) ** 2)

    return np.nanmean(err)

def cross_validation(votes, dim, n_folds=5, verbose=False):
    folds = k_fold(n_folds, votes)
    err = []
    for i in range(n_folds):
        err.append(unit_test(i, folds, dim))
    if (verbose): print(err)
    return np.mean(err)

#EndRegion

#Region[Magenta] Dimension choice

def dim_test(votes, n_folds=10, verbose=False, dim_min=5, dim_max=20):
    err_min = np.PINF
    dim = None
    for k in range(dim_min, dim_max):
        err = cross_validation(votes, k)
        if verbose:
            print("Erreur pour k = "+str(k)+" : "+str(err))
        if err < err_min:
            err_min = err
            dim = k
    return dim, err

#EndRegion