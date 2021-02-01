import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

from scripts.cross_valid import cross_validation

#Region[Blue] Init : read csv data

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

#EndRegion

#Region[Cyan] Question 1

result_q1 = cross_validation(votes, users, items, 10, False)
print(result_q1)
#EndRegion