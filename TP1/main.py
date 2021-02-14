import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from scripts.cross_valid import cross_validation
from scripts.item_item import cos_matrix, votes_communs, nb_voisins, compute_votes_manquants
from scripts.svd import dim_test, svd_cross_validation
from scripts.agglomeration import agglomeration
from scripts.item_item_v2 import predict_i, cos_matrix, closest_neighbor, compute_K, compute_mean_R_0
import scripts.svd as d

'''
#######################################################################
#   RESULTATS A AFFICHER : PARAMATERES DE MAIN (QU1, QU2, QU3, QU4)   #
#######################################################################
'''

def main(QU1=False, QU2=True, QU3=False, QU4=False):

    plt.ion()

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

    # Region[Cyan] Question 1

    if QU1:
        print("Q1. Calcul de la MSE de l'approche moyenne pour établir un seuil de comparaison (validation croisée, 10 replis)")
        result_q1 = cross_validation(votes, 10, False)
        print("Erreur quadratique moyenne : " + str(result_q1))
        print("- - -")

    # EndRegion

    # Region[Cyan] Question 2

    if QU2:
        cos = cos_matrix(model)
        ind = closest_neighbor(cos, 10)
        w = cos[np.arange(len(cos)), ind]

        mean, R_0 = compute_mean_R_0(model)
        res = np.zeros(model.shape)
        for iid in range(1): #range(model.shape[1]):
            ind_i = ind[:, iid]
            w_i = w[:, iid]
            v_0_i = R_0[:, ind_i]
            K_i = compute_K(v_0_i,w_i)
            res[:,iid] = predict_i(mean[iid], K_i, v_0_i, w_i)

        """cos_matrix = cos_matrix(model)
        triangular_cos = np.tril(cos_matrix)

        vote_comm = votes_communs(model)
        voisins = nb_voisins(vote_comm)

        # Question 2.a.

        plt.hist(triangular_cos[triangular_cos > 0], color='cyan', edgecolor='black', bins=250)
        plt.title("Q2.a. Distribution des similarités")
        plt.ylabel("Nombre d'occurence")
        plt.xlabel("Valeur de cosinus")
        plt.show(block=True)

        p_zeros = 1 - np.count_nonzero(cos_mat) / (2 * n_items * (n_items + 1) / 2)
        print("Q2.a. Proportion de poids nuls : " + str(p_zeros * 100)[:4] + " %")

        # Question 2.b.

        plt.hist(voisins, bins=168)
        plt.title("Q2.b. Distribution du nombre de voisin avec vote commun par item")
        plt.ylabel("Nombre d'occurence")
        plt.xlabel("Nombre de voisin")
        plt.show(block=True)"""

    #EndRegion

    # Region[Cyan] Question 3
    
    if QU3:
        print("Q3. Choix des dimensions à garder pour SVD - on minimise l'erreur par cross-validation avec 5 replis")

        dim_min, dim_max = 5, 20
        dim, err, err_list = dim_test(votes, dim_min, dim_max)

        plt.plot(range(dim_min, dim_max+1), err_list,'d-r')
        plt.title("Erreur quadratique moyenne de l'approche SVD pour différents choix de dimensions")
        plt.ylabel("MSE")
        plt.xlabel("k")
        plt.show(block=True)
        
        print("On choisit de garder "+str(dim)+" dimensions, et on a alors une erreur de "+str(err))
        print("- - -")

    #EndRegion

    # Region[Cyan] Question 4

    if QU4:
        print("Q4. Calcul de MSE pour différentes tailles de classes pour l'approche par agglomération")
        cluster_size = [5,10,20,40,80]
        for size in cluster_size:
            err = agglomeration(votes, 5, size)
            print("La MSE avec 5 replis et " + str(size) + " clusters est de " + str(err))
        print("- - -")

    #EndRegion

    #print(d.svd_cross_validation(votes, 14, 5, True))

main(False, False, True, False)