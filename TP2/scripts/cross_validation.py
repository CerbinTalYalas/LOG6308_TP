import pandas as pd
import numpy as np
import random
from scipy.sparse import csr_matrix

adjacent = pd.read_table("./data/citeseer.rtable", sep=" ")
adjacent.columns = adjacent.columns.astype('int64')

def gen_sets(adjacent, k=10):
    p = 1/k
    adjacent_values = adjacent.values
    l, c = np.where(adjacent_values == 1)
    train_matrix = np.ndarray.copy(adjacent_values)
    i = len(l)
    sets=[]
    test_indexes = list(range(i))
    random.shuffle(test_indexes)
    fold_size = int(i/k)
    for fold in range(k):
        df_train = adjacent.copy()
        df_test = pd.DataFrame(columns=adjacent.columns, index=adjacent.index, data=0.0)
        fold_indexes = test_indexes[fold*fold_size : (fold+1)*fold_size]
        for index in fold_indexes:
            test = (df_train.index[l[index]], df_train.columns[c[index]])
            df_train.at[test[0], test[1]] = 0
            df_test.at[test[0], test[1]] = 1
        sets.append((df_train, df_test))
    return sets

def fn_taux_rappel(df_adjacence_test, recommandations):

    mx_test = df_adjacence_test.values.astype(bool)
    mx_reco = recommandations.values.astype(bool)

    recall = mx_test & mx_reco # Link is in test set AND is predicted

    recall_rate = np.sum(recall)/np.sum(mx_test)

    return recall_rate

def tx_rappel_total(df_adjacence, n=5, training_func):
    
    recall_mx = []

    for _ in range(n):
        exp = []
        sets = gen_sets(df_adjacence)
        for df_adjacence_entrainement, df_adjacence_test in sets:
            recommandations = training_func(df_adjacence_entrainement)
            recall = fn_taux_rappel(df_adjacence_test, recommandations)
            exp.append(recall)
        recall_mx.append(exp)

    return np.mean(recall_mx)

