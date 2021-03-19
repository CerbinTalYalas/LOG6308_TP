import pandas as pd
import numpy as np
import random
from scipy.sparse import csr_matrix

adjacent = pd.read_table("./data/citeseer.rtable", sep=" ")
adjacent.columns = adjacent.columns.astype('int64')

def gen_sets(adjacent, p=0.1):
    df_train = adjacent.copy()
    df_test = pd.DataFrame(columns=adjacent.columns, index=adjacent.index, data=0.0)
    adjacent_values = adjacent.values
    l, c = np.where(adjacent_values == 1)
    train_matrix = np.ndarray.copy(adjacent_values)
    i = len(l)
    test_indexes = random.sample(range(i), int(p*i))
    for index in test_indexes:
        test = (df_train.index[l[index]], df_train.columns[c[index]])
        df_train.at[test[0], test[1]] = 0
        df_test.at[test[0], test[1]] = 1
    return df_train, df_test

def fn_taux_rappel(df_test, recommandations):

    mx_test = df_test.values.astype(bool)
    mx_reco = recommandations.values.astype(bool)

    recall = mx_test & mx_reco # Link is in test set AND is predicted

    recall_rate = np.sum(recall)/np.sum(mx_test)

    return recall_rate