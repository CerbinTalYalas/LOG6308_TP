import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from scripts.cross_valid import cross_validation
from scripts.svd import dim_test
from scripts.agglomeration import agglomeration
from scripts.item_item import cos_matrix, closest_neighbor, votes_communs, nb_voisins, compute_error

'''
#######################################################################
#   RESULTATS A AFFICHER : PARAMATERES DE MAIN (QU1, QU2, QU3, QU4)   #
#######################################################################
'''


def main(QU1=False, QU2=True, QU3=True, QU4=False):
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
        print(
            "Q1. Calcul de la MSE de l'approche moyenne pour établir un seuil de comparaison (validation croisée, 10 replis)")
        result_q1 = cross_validation(votes, 10, False)
        print("Erreur quadratique moyenne : " + str(result_q1))
        print("- - -")

    # EndRegion

    # Region[Cyan] Question 2

    if QU2:
        cos = cos_matrix(model)
        ind = closest_neighbor(cos, 10)
        w = cos[np.arange(len(cos)), ind]

        # Question 2.a.

        plt.hist(w.flatten(), bins=250, color='cyan', edgecolor='black')
        plt.title("Q2.a. Distribution des similarités parmi les 10 voisins")
        plt.ylabel("Nombre d'occurence")
        plt.xlabel("Valeur de w")
        plt.show(block=True)

        nb_0 = np.count_nonzero(w == 0)
        p_zeros = nb_0 / w.size
        print(
            "Q2.a. Proportion moyenne de poids nuls parmi les 10 voisins d'un item : " + str(p_zeros * 100)[:4] + " %")

        nb_0_g = np.count_nonzero(cos == 0)
        p_zeros_g = nb_0_g / cos.size
        print("Q2.a. Proportion moyenne de poids nuls parmi tous les voisins d'un item : " + str(p_zeros_g * 100)[
                                                                                             :4] + " %")

        # Question 2.b.

        vote_comm = votes_communs(model)
        voisins = nb_voisins(vote_comm)

        
        plt.hist(voisins, bins=168, color='cyan', edgecolor='black')
        plt.title("Q2.b. Distribution du nombre de voisin avec vote commun par item")
        plt.ylabel("Nombre d'occurence")
        plt.xlabel("Nombre de voisin")
        
        plt.show(block=True)

        pl_voisins_vote_commun = []
        for uid in range(model.shape[0]):
            for iid in range(model.shape[1]):
                nvoisins = voisins[iid]
                nusers = np.count_nonzero(model[uid])-1*(model[uid,iid] != 0)
                pl_voisins_vote_commun.append((nvoisins-nusers)/nvoisins)

        p_voisins_vote_commun = np.mean(pl_voisins_vote_commun)

        print("Q2.b. Proportion moyenne de votes manquants parmi tous les voisins d'un item : " + str(p_voisins_vote_commun * 100)[
                                                                                             :4] + " %")


        # Question 2.d.

        err, mean_mse = compute_error(model, ind, w, votes)

        print("Q2.d. L'erreur quadratique moyenne pour l'approche item-item est de : " + str(mean_mse))

        plt.hist(err, bins=100, color='cyan', edgecolor='black')
        plt.title("Q2.d. Distribution de l'erreur quadratique moyenne par item")
        plt.ylabel("Nombre d'occurence")
        plt.xlabel("Valeur de MSE")
        plt.show(block=True)
        print("- - -")

    # EndRegion

    # Region[Cyan] Question 3

    if QU3:
        print("Q3. Choix des dimensions à garder pour SVD - on minimise l'erreur par cross-validation avec 5 replis")

        dim_min, dim_max = 5, 20
        dim, err, err_list = dim_test(votes, dim_min, dim_max)

        plt.plot(range(dim_min, dim_max + 1), err_list, 'd-r')
        plt.title("Erreur quadratique moyenne de l'approche SVD pour différents choix de dimensions")
        plt.ylabel("MSE")
        plt.xlabel("k")
        plt.show(block=True)

        print("On choisit de garder " + str(dim) + " dimensions, et on a alors une erreur de " + str(err))
        print("- - -")

    # EndRegion

    # Region[Cyan] Question 4

    if QU4:
        print("Q4. Calcul de MSE pour différentes tailles de classes pour l'approche par agglomération")
        cluster_size = [5, 10, 20, 40, 80]
        for size in cluster_size:
            err = agglomeration(votes, 5, size)
            print("La MSE avec 5 replis et " + str(size) + " clusters est de " + str(err))
        print("- - -")

    # EndRegion


main(True, True, True, True)