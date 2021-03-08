import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# Region[Blue] Init : read csv data

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

model = csr_matrix((votes['rating'], (votes['user.id'], votes['item.id']))).toarray()

# EndRegion