import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scripts.bayes_pred import compute_error, get_k_fav_item, compute_average_votes
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

model = csr_matrix((votes['rating'], (votes['user.id'], votes['item.id']))).toarray()

# EndRegion

def main(Q1=True, Q2=False, Q3=False, Q4=False, Q5=False):

#Region[Yellow] Q1

    if Q1:

        print("10 prédictions pour une femme ingénieure de 23 ans :")
        features = ('engineer', 'F', 'young')
        topk = get_k_fav_item(features, items, votes, users)
        for entry in topk:
            odd, film = entry
            print("| "+str(film[' movie title '])+" - Odd ratio : "+str(odd)[0:4])
        print("\n - - - \n")
        avg_votes = compute_average_votes(model)
        err = compute_error(model, votes, users, avg_votes)
        print("MSE sur les votes prédits : "+str(err)[0:5])

#EndRegion

main()