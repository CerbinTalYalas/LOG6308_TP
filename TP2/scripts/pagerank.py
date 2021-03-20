import pandas as pd
import numpy as np


def compute_pagerank(adjacent, d=0.85, e=1e-6):
    adjacent_temp = adjacent.copy()
    np.fill_diagonal(adjacent_temp.values, 1)
    adjacent_temp[adjacent_temp > 1] = 1

    n = adjacent_temp.shape[0]
    r = pd.Series(1 / n, index=adjacent_temp.index)
    s = adjacent_temp.sum(axis=1)
    adjacent_t = adjacent_temp.transpose()

    last_r = r.copy()
    convergence = False
    while not convergence:
        r = (1 - d) / n + d * adjacent_t.dot(r / s)
        norm = np.linalg.norm((last_r - r).to_numpy())
        if norm < e: convergence = True
        last_r = r.copy()

    return r


def get_doc_top_pagerank(doc, adjacent, pagerank, top_n=10):
    adjacent[adjacent > 1] = 1
    doc_ref = adjacent.loc[doc]
    local_pagerank = doc_ref * pagerank
    result = local_pagerank[local_pagerank > 0]
    result = result.nlargest(top_n)
    return result

def get_top_pagerank(adjacent, pagerank, top_n=10):
    adjacent_copy = adjacent.copy()
    #adjacent_copy[adjacent_copy > 1] = 1
    adjacent_pagerank = adjacent * pagerank
    result = adjacent_pagerank.apply(lambda s, n: pd.Series(s.nlargest(n).index), axis=1, n=top_n)
    return result

