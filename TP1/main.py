import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from scripts.cross_valid import cross_validation
from scripts.item_item import cos_matrix, vote_commun, nb_voisins
from scripts.svd import dim_test
from scripts.agglomeration import agglomeration

QUESTION = [False, True, False, False]

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

if QUESTION[0]:
# Region[Cyan] Question 1

    print("Q1. Calcul de la MSE pour un seuil de comparaison (validation croisée, 10 replis)")
    result_q1 = cross_validation(votes, 10, False)
    print("Erreur quadratique moyenne : " + str(result_q1))
    print("- - -")

# EndRegion

if QUESTION[1]:
    # Region[Cyan] Question 2

    cos_matrix = cos_matrix(model)
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
    vote = vote_commun(model)
    voisin = nb_voisins(vote)

    plt.hist(voisin, bins=168)
    plt.title("Q2.b. Distribution du nombre de voisin avec vote commun par item")
    plt.ylabel("Nombre d'occurence")
    plt.xlabel("Nombre de voisin")
    plt.show()


if QUESTION[2]:
    print("Q3. Choix des dimensions à garder pour SVD - on minimise l'erreur par cross-validation avec 5 replis")
    dim, err = dim_test(votes)
    print("On choisit de garder "+str(dim)+" dimensions, et on a alors une erreur de "+str(err))
    print("- - -")


if QUESTION[3]:
    print("Q4. Calcul de MSE pour différentes tailles de classes pour l'approche par agglomération")
    cluster_size = [5,10,20,40,80]
    for size in cluster_size:
        err = agglomeration(votes, 5, size)
        print("La MSE avec 5 replis et " + str(size) + " clusters est de " + str(err))
    print("- - -")
