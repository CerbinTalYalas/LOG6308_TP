import pandas as pd
import numpy as np

adjacent = pd.read_table("./citeseer.rtable", sep=" ")

adjacent.columns = adjacent.columns.astype('int64')
np.fill_diagonal(adjacent.values, 0)
doc = 422908


def compute_pagerank(adjacent, d=0.85):
    adjacent_temp = adjacent.copy()
    np.fill_diagonal(adjacent_temp.values, 1)
    adjacent_temp[adjacent_temp > 1] = 1

    n = adjacent_temp.shape[0]
    r = pd.Series(1, index=adjacent_temp.index)
    s = adjacent_temp.sum(axis=1)
    adjacent_t = adjacent_temp.transpose()

    last_rsum = 10000
    convergence = False
    while not convergence:
        r = (1 - d) / n + d * adjacent_t.dot(r / s)
        r_sum = r.sum()
        if last_rsum - r_sum < 0.01: convergence = True
        last_rsum = r_sum

    return r


def get_top_pagerank(doc, adjacent, pagerank):
    doc_ref = adjacent.loc[doc]
    adjacent[adjacent > 1] = 1
    local_pagerank = doc_ref * pagerank
    result = local_pagerank[local_pagerank>0]
    result.sort_values(ascending=False, inplace=True)
    #top_page = local_pagerank.nlargest(top)
    return result


pagerank = compute_pagerank(adjacent)
result1 = get_top_pagerank(doc, adjacent, pagerank)
print(result1)

adjacent2 = adjacent + adjacent.dot(adjacent)
result2 = get_top_pagerank(doc, adjacent2, pagerank)
print(result2)