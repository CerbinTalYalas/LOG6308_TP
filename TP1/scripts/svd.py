import numpy as np
import pandas as pd
import scipy as sp
import math
from scipy.sparse import csr_matrix
from scipy.linalg import sqrtm

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
    return u, np.diag(s), vh

def svd_decomp_reduc(model, k):
    U, s, V = svd_decomp(model)
    s=s[0:k,0:k]
    U=U[:,0:k]
    V=V[0:k,:]
    
    UsV = U @ s @ V

    return UsV

#EndRegion

#Region[Green] Cross validation

def train(data_train, k):
    train_model = csr_matrix((data_train['rating'], (data_train['user.id'], data_train['item.id'])), shape=(943, 1682), dtype=float).toarray()
    
    mask = (train_model == 0.0)
    masked_arr = np.ma.masked_array(train_model, mask)
    item_means = np.mean(masked_arr, axis=0, dtype=float).filled(0.0)    # nan entries will replaced by the average rating for each item
    user_means = np.mean(masked_arr, axis=1, dtype=float).filled(0.0)
    average_means=(1/2*(item_means+user_means[:,np.newaxis]))

    centered = np.array(masked_arr.filled(average_means))-average_means

    svd_matrix = svd_decomp_reduc(centered, k)+average_means
    return svd_matrix

def unit_test(test_index, folds, k):
    test_set = folds[test_index]
    train_set = pd.concat(folds[:test_index]+folds[test_index + 1:])

    model = train(train_set, k)

    err = []
    for _, vote in test_set.iterrows():
        err.append((vote['rating'] - model[vote['user.id'],vote['item.id']]) ** 2)

    return np.nanmean(err)

def svd_cross_validation(votes, dim, n_folds=5, verbose=False):
    folds = k_fold(n_folds, votes)
    err = []
    for i in range(n_folds):
        err.append(unit_test(i, folds, dim))
    if (verbose): print(err)
    return np.mean(err)

#EndRegion

#Region[Magenta] Dimension choice

def dim_test(votes, dim_min, dim_max, n_folds=10, verbose=False):
    err_min = np.PINF
    dim = None
    err_list = []
    for k in range(dim_min, dim_max+1):
        err = svd_cross_validation(votes, k)
        err_list.append(err)
        if verbose:
            print("Erreur pour k = "+str(k)+" : "+str(err))
        if err < err_min:
            err_min = err
            dim = k
    return dim, err, err_list

#EndRegion

def compar_matrix(votes, k, model):
    v = train(votes, k)
    return((model-v)**2)