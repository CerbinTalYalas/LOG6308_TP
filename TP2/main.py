import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import time

from scripts.mx_bayes_pred import compute_predictions, process_error, get_k_fav_item
from scripts.pagerank import compute_pagerank, get_top_pagerank
from scripts.cos_similarity import compute_cos_similarity, get_doc_recommendation

# Region[Blue] Init : read csv data


### Q1 data
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

model = csr_matrix((votes['rating'], (votes['user.id'], votes['item.id']))).toarray() # Row : user / Col : item

### Q2-5 data

adjacent = pd.read_table("./data/citeseer.rtable", sep=" ")
adjacent.columns = adjacent.columns.astype('int64')
np.fill_diagonal(adjacent.values, 0)
pagerank = compute_pagerank(adjacent)

# EndRegion

def main(Q1=True, Q2=True, Q3=True, Q4=True, Q5=False):

#Region[Yellow] Q1

    if Q1:
        print("\n ### QUESTION 1 ###\n")
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
        print("\n ### QUESTION 2 ###\n")
        doc = 422908

        result1 = get_top_pagerank(doc, adjacent, pagerank)
        print("Lectures recommandées après la lecture "+str(doc)+" - approche de base")
        for iid, score in result1.iteritems():
            print("|  Id : "+str(iid)+" - Pagerank : "+str(score)[0:7])
        print("- - -")
        adjacent_ext = adjacent + adjacent.dot(adjacent)
        result2 = get_top_pagerank(doc, adjacent_ext, pagerank)
        print("Lectures recommandées après la lecture "+str(doc)+" - voisinage étendu")
        for iid, score in result2.iteritems():
            print("|  Id : "+str(iid)+" - Pagerank : "+str(score)[0:7])

    if Q3:
        print("\n ### QUESTION 3 ###\n")
        all_result = compute_cos_similarity(adjacent)
        print("10 articles les plus similaires à l'article "+str(doc)+" :")
        reco = get_doc_recommendation(all_result, doc)
        for rank, iid in reco.iteritems():
            print("| Id : "+str(iid)+" - Classement : "+str(rank))


#EndRegion

main(True, False, False, False, False)