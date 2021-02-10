import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from scripts.cross_valid import cross_validation
#from item_item import ii_cos_matrix
from scripts.agglomeration import agglomeration_1, agglomeration_2

QUESTION = "Q4"

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

if QUESTION == "Q1" or QUESTION == "ALL":
    # Region[Cyan] Question 1

    result_q1 = cross_validation(votes, users, items, 10, False)
    print("Q1. Erreur quadratique moyenne : " + str(result_q1))

# EndRegion

"""if QUESTION == "Q2" or QUESTION == "ALL":
    # Region[Cyan] Question 2

    cos_matrix = ii_cos_matrix(model)
    triangular_cos = np.tril(cos_matrix)

    # Question 2.a.

    plt.hist(triangular_cos[triangular_cos > 0], color='cyan', edgecolor='black', bins=250)
    plt.title("Q2.a. Distribution des similarité")
    plt.ylabel("Nombre d'occurence")
    plt.xlabel("Valeur de cosinus")
    plt.show()

    p_zeros = 1 - np.count_nonzero(cos_matrix) / (2 * n_items * (n_items + 1) / 2)
    print("Q2.a. Proportion de poids nuls : " + str(p_zeros * 100)[:4] + " %")

    # Question 2.b.

# EndRegion"""

if QUESTION == "Q4" or QUESTION == "ALL":
    cluster_size = [5,10,20,40,80]
    for size in cluster_size:
        err = agglomeration_1(votes, 5, size)
        print("La mse avec 5 replis et " + str(size) + " clusters est de " + str(err))