import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scripts.mx_bayes_pred import compute_predictions, process_error, get_k_fav_item
import time

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

# Row : user / Col : item
model = csr_matrix((votes['rating'], (votes['user.id'], votes['item.id']))).toarray()

# EndRegion

def main(Q1=True, Q2=True, Q3=False, Q4=False, Q5=False):

#Region[Yellow] Q1

    if Q1:
        print('\n ### QUESTION 1 ###\n')
        print("10 prédictions pour une femme ingénieure de 23 ans :")
        features = ('engineer', 'F', 'young')
        topk = get_k_fav_item(features, model, votes, users, items)
        for entry in topk:
            rating, iid = entry
            film = items.iloc[iid]
            print("| "+str(film[' movie title '])+" - Note prédite : "+str(rating)[0:5])
        print("- - -")
        pred = compute_predictions(model, votes, users, items)
        mse = process_error(model, pred)
        print("MSE sur les votes prédits : "+str(mse)[0:5])

    if Q2:
        print('\n ### QUESTION 2 ###\n')

#EndRegion

main()