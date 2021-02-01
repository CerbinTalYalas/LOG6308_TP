import pandas as pd
import numpy as np
import scipy as sp
import math
import warnings
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

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

#Region[Yellow] Item_item

def ii_cos(i1, i2):
    return np.dot(i1, i2)/(np.linalg.norm(i1)*np.linalg.norm(i2))

def ii_eval_matrix(eval, model):
    nb_items = len(model[0])
    val_matrix = np.NINF*np.ones((nb_items, nb_items))
    for i1 in range(nb_items):
        for i2 in range(i1+1, nb_items):
            value = eval(model[:,i1], model[:,i2])
            val_matrix[i1,i2] = value
            val_matrix[i2,i1] = value
    return val_matrix

def ii_closest_neighbor(dist_items, k):
    return np.argpartition(dist_items, -k)[-k:]

def ii_predict(uid, iid, model, k, dist_matrix=ii_eval_matrix(ii_cos, model)):
    neighbors = ii_closest_neighbor(dist_matrix[iid], k)
    w = dist_matrix[iid]

    predict = np.mean(model[:, iid][model[:, iid] > 0])
    k,s = 0,0
    for inid in neighbors:
        k += abs(w[inid])
        s += w[inid]*(model[uid, inid]-np.mean(model[:, inid][model[:, inid] > 0]))
    predict += 1/k*s
    return predict

#EndRegion

model = csr_matrix((votes['rating'], (votes['user.id'], votes['item.id']))).toarray()
print(ii_predict(300,97,model,10))