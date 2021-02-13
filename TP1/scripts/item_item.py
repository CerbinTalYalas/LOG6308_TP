import pandas as pd
import numpy as np
import scipy as sp
import math
import warnings
import time
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

#Region[Yellow] Item_item prediction

def cos_matrix(model):
    cos_matrix = 1.0*np.matmul(model.transpose(),model)
    nb_items = len(model[0])
    for i in range(nb_items):
        norm = np.linalg.norm(model[:,i])
        cos_matrix[i] = cos_matrix[i]/norm
        cos_matrix[:,i] = cos_matrix[:,i]/norm
        cos_matrix[i,i] = 1.0*np.NINF
    return cos_matrix

def closest_neighbor(dist_items, k):
    return np.argpartition(dist_items, -k)[-k:]

def predict(uid, iid, model, k, dist_matrix, verbose = True):
    neighbors = closest_neighbor(dist_matrix[iid], k)
    w = dist_matrix[iid][neighbors]

    if verbose:
        w_analysis(w)

    v_n = model[uid][neighbors]
    v_m = np.array(list(np.mean(model[:, n][model[:, n] > 0]) for n in neighbors))
    s = v_n - v_m

    predict = np.mean(model[:, iid][model[:, iid] > 0])+(1/np.linalg.norm(w, ord=1))*np.matmul(w.transpose(), s)
    return max(1,min(5,predict))

#EndRegion

#Region[DGray] item-item analysis

def w_analysis(w):

    plt.hist(w[w>0], color='cyan', edgecolor = 'black', bins= 250)
    plt.title("Distribution des similarités")
    plt.ylabel("Nombre d'occurence")
    plt.xlabel("Valeur de cosinus")
    plt.show()

    p_zeros = 1-np.count_nonzero(w)/len(w)
    print("Proportion de poids nuls parmi les voisins : "+str(p_zeros*100)[:4]+" %")

def votes_communs(model):
    model_vote = np.ma.masked_where(model != 0, model).mask
    model_vote = model_vote.astype(int)
    nb_vote_commun = np.matmul(model_vote.transpose(), model_vote)
    np.fill_diagonal(nb_vote_commun, 0)
    return nb_vote_commun

def nb_voisins(data):
    voisin = np.ma.masked_where(data != 0, data).mask
    return voisin.sum(axis=0) #2a

def prop_votes_manquants(uid, iid, model, voisins=None):
    if voisins is None : voisins = nb_voisins(votes_communs(model))
    votes_user = np.count_nonzero(model[uid])
    return (voisins[iid]-(votes_user-1))/voisins[iid]

def compute_votes_manquants(votes_pd, model):
    voisins = nb_voisins(votes_communs(model))
    res = np.zeros(model.shape)
    for _, vote in votes_pd.iterrows():
        uid = vote["user.id"]
        iid = vote["item.id"]
        res[uid,iid] = prop_votes_manquants(uid, iid, model, voisins)
    return res

#EndRegion