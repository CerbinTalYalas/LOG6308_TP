import pandas as pd
import numpy as np
import os
import heapq
import time
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

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

#Region[Yellow] Classifying functions

def like(rating):
    return rating > 3

def age_group(age):
    if age > 25:
        return 'old'
    return 'young'

def get_features(user):
    job = user[' job ']
    gender = user[' gender ']
    age_grp = age_group(user[' age '])
    return job, gender, age_grp

def extended_votes(votes, users):
    xt_votes = pd.merge(votes, users, how='inner', left_on='user.id', right_on='id ')

    xt_votes['like'] = xt_votes['rating'].apply(like)
    xt_votes['age_group'] = xt_votes[' age '].apply(age_group)
    xt_votes.rename(columns = {' job ':'job', ' gender ':'gender'}, inplace=True)

    xt_votes = xt_votes[['item.id', 'like', 'job', 'gender', 'age_group']]
    xt_votes.sort_values(by=['item.id', 'job', 'gender', 'age_group'], inplace=True)

    return xt_votes

#EndRegion

#Region[Green] Laplace Correction and Odd ratio

def laplace_correction(t, b):
    if t==0 or b==0:
        return (t+1)/(b+2)
    return t/b
v_laplace = np.vectorize(laplace_correction)

def odd_compute(like, slike, dislike, sdislike):
    if like==0 and dislike==0:
        return 1.0
    else:
        return laplace_correction(like, slike)/laplace_correction(dislike, sdislike)
v_odd = np.vectorize(odd_compute)

#EndRegion

#Region[Cyan] Bayes prediction


def gen_dataframe(votes_df, n_items, feature):

    index_df = pd.DataFrame(np.arange(n_items), columns=['item.id'])
    data_df = votes_df[['item.id',feature]].value_counts(sort=False).unstack()
    data_df.reset_index(inplace=True)
    temp_df = index_df.merge(data_df, on='item.id', how='outer')

    sparse_df = temp_df.drop('item.id', axis=1)
    sparse_df = sparse_df.fillna(0.0)

    return sparse_df

def compute_odd_ratio(feat_like, feat_dislike):
    odd_data = []
    odd_col = feat_like.columns
    for i,_ in feat_like.iterrows():
        row_like = feat_like.iloc[i]
        s_like = row_like.sum()
        row_dislike = feat_dislike.iloc[i]
        s_dislike = row_dislike.sum()
        odd_data.append(v_odd(row_like, s_like, row_dislike, s_dislike))

    odd_df = pd.DataFrame(data=odd_data, columns=odd_col)
    return odd_df


def gen_bayes_ls(xt_votes, n_items):
    votes_like, votes_dislike = xt_votes[xt_votes['like'] == True], xt_votes[xt_votes['like'] == False]

    job_like, job_dislike = gen_dataframe(votes_like, n_items, 'job'), gen_dataframe(votes_dislike, n_items, 'job')
    ls_job = compute_odd_ratio(job_like, job_dislike)
    age_like, age_dislike = gen_dataframe(votes_like, n_items, 'age_group'), gen_dataframe(votes_dislike, n_items, 'age_group')
    ls_age = compute_odd_ratio(age_like, age_dislike)
    gender_like, gender_dislike = gen_dataframe(votes_like, n_items, 'gender'), gen_dataframe(votes_dislike, n_items, 'gender')
    ls_gender = compute_odd_ratio(gender_like, gender_dislike)

    return ls_job, ls_age, ls_gender


def item_votes_analysis(model):
    like_ratings = []
    dislike_ratings = []
    votes_ratio = []
    for item in model.T:
        likes = np.where(item>3, item, np.nan)
        if np.isnan(likes).all():
            mr_likes = 0
            n_lvotes = 0
        else:
            mr_likes = np.nanmean(likes)
            n_lvotes = np.count_nonzero(~np.isnan(likes))
        dislikes = np.where(((item > 0) & (item <= 3)), item, np.nan)
        if np.isnan(dislikes).all():
            mr_dislikes = 0
            n_dvotes = 0
        else:
            mr_dislikes = np.nanmean(dislikes)
            n_dvotes = np.count_nonzero(~np.isnan(dislikes))
        like_ratings.append(mr_likes)
        dislike_ratings.append(mr_dislikes)
        votes_ratio.append(laplace_correction(n_lvotes,n_dvotes))
    return np.array(like_ratings), np.array(dislike_ratings), np.array(votes_ratio)

#EndRegion

#Region[Red] Compute vote and generate predicted model

# Odd to Probability
def otp(o):
    return o/(1+o)
vect_otp = np.vectorize(otp)

def compute_predictions(model, votes, users, items):

    xt_votes = extended_votes(votes, users)
    like_ratings, dislike_ratings, votes_ratio = item_votes_analysis(model)
    ls_job, ls_age, ls_gender = gen_bayes_ls(xt_votes, len(items))

    pred_model = []

    for _,user in users.iterrows():
        job, gender, age_grp = get_features(user)

        v_job = np.array(ls_job[job])
        v_age = np.array(ls_age[age_grp])
        v_gender = np.array(ls_gender[gender])

        odd_ratio = votes_ratio*v_job*v_age*v_gender
        prob_user = vect_otp(odd_ratio)
        pred_user = prob_user*like_ratings + (1-prob_user)*dislike_ratings

        pred_model.append(pred_user)

    return np.array(pred_model)

#EndRegion

#Region[DGray] Top k prediction

def get_k_fav_item(features, model, votes, users, items, k=10):
    topk = []
    
    job, gender, age_grp = features

    xt_votes = extended_votes(votes, users)
    like_ratings, dislike_ratings, votes_ratio = item_votes_analysis(model)
    ls_job, ls_age, ls_gender = gen_bayes_ls(xt_votes, len(items))

    v_job = np.array(ls_job[job])
    v_age = np.array(ls_age[age_grp])
    v_gender = np.array(ls_gender[gender])

    odd_ratio = votes_ratio*v_job*v_age*v_gender
    prob_user = vect_otp(odd_ratio)
    pred_user = prob_user*like_ratings + (1-prob_user)*dislike_ratings

    topk = sorted(((value, index) for index, value in enumerate(pred_user)), reverse=True)[:k]
    
    return topk

#EndRegion

#Region[Magenta] Compute error on model

def process_error(model, prediction):

    error = (model-prediction)**2
    masked = np.ma.masked_where(model == 0, error)

    mse = np.nanmean(masked)

    return mse

#EndRegion