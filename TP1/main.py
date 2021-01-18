import pandas as pd
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
#Region[Blue] Init

users = pd.read_csv("./data/u.csv", sep="|")
n_users = len(users)
users['id '] -= 1
items = pd.read_csv("./data/items.csv", sep="|")
n_items = len(items)
items['movie id '] -= 1
votes = pd.read_csv("./data/votes.csv", sep="|")
n_votes = len(votes)
votes['user.id'] -= 1
votes['item.id'] -= 1

m_votes = csr_matrix((votes['rating'], (votes['user.id'], votes['item.id']))).toarray()

#EndRegion

#Region[Green] Average votes prediction

avg_votes = np.zeros(m_votes.shape)
avg_votes[1,1] = 1

def avg_calc(r, c, m):
    avg_r = np.mean( m[r][m[r] > 0] )
    avg_c = np.mean( m[:, c][m[:, c]>0] )
    return (avg_r+avg_c)/2

for u in range(n_users):
    for i in range(n_items):
        avg_votes[u,i] = avg_calc(u,i,m_votes)

print(avg_votes)

#EndRegion