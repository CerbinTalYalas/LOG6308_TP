import pandas as pd
import numpy as np
import random
from scipy.sparse import csr_matrix

adjacent = pd.read_table("./data/citeseer.rtable", sep=" ")
adjacent.columns = adjacent.columns.astype('int64')

def gen_sets(adjacent_values, p=0.1):
    l, c = np.where(adjacent_values == 1)
    train_matrix = np.ndarray.copy(adjacent_values)
    i = len(l)
    test_indexes = random.sample(range(i), int(p*i))
    test_set = []
    for index in test_indexes:
        test = (l[index], c[index])
        test_set.append(test)
        train_matrix[test] = 0
    return train_matrix, test_set

tr, te = gen_sets(adjacent.values)
print(tr)
print(len(te))