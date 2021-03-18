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


def get_top_pagerank(doc, adjacent, pagerank):
    doc_ref = adjacent.loc[doc]
    adjacent[adjacent > 1] = 1
    local_pagerank = doc_ref * pagerank
    result = local_pagerank[local_pagerank > 0]
    result.sort_values(ascending=False, inplace=True)
    # top_page = local_pagerank.nlargest(top)
    return result
