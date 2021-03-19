import pandas as pd
import numpy as np
import random
from scipy.sparse import csr_matrix

adjacent = pd.read_table("./data/citeseer.rtable", sep=" ")
adjacent.columns = adjacent.columns.astype('int64')

def gen_sets(adjacent, p=0.1):
    adjacent_train = adjacent.copy()
    adjacent_values = adjacent.values
    l, c = np.where(adjacent_values == 1)
    train_matrix = np.ndarray.copy(adjacent_values)
    i = len(l)
    test_indexes = random.sample(range(i), int(p*i))
    test_set = []
    for index in test_indexes:
        test = (adjacent_train.index[l[index]], adjacent_train.columns[c[index]])
        test_set.append(test)
        adjacent_train.at[l[index], c[index]] = 0
    return adjacent_train, test_set


tr, te = gen_sets(adjacent)
print(tr)
print(te)