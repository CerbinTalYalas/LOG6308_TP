import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from scripts.cross_valid import cross_validation
import scripts.item_item as ii
import scripts.svd as svd

QUESTION = "Q0"

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

model = csr_matrix((votes['rating'], (votes['user.id'], votes['item.id']))).toarray()

#EndRegion

if QUESTION == "Q1" or QUESTION == "ALL":

#Region[Cyan] Question 1

    result_q1 = cross_validation(votes, users, items, 10, False)
    print("Q1. Erreur quadratique moyenne : "+str(result_q1))

#EndRegion

if QUESTION == "Q2" or QUESTION == "ALL":

#Region[Cyan] Question 2

    cos_matrix = ii.cos_matrix(model)
    triangular_cos = np.tril(cos_matrix)
    
    # Question 2.a.

    """ plt.hist(triangular_cos[triangular_cos>0], color='cyan', edgecolor = 'black', bins= 250)
    plt.title("Q2.a. Distribution des similarité")
    plt.ylabel("Nombre d'occurence")
    plt.xlabel("Valeur de cosinus")
    plt.show()

    p_zeros = 1-np.count_nonzero(cos_matrix)/(2*n_items*(n_items+1)/2)
    print("Q2.a. Proportion de poids nuls : "+str(p_zeros*100)[:4]+" %") """

    ii.predict(10,10,model,10,cos_matrix)

    # Question 2.b.

#EndRegion

print(svd.square_error(model, 10))

#Region[Cyan] Question 3