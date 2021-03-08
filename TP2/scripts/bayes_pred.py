import pandas as pd
import numpy as np
import os
import heapq
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

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

#Region[Cyan] Bayes prediction

def count_likes(row, feature, xt_votes):
    item_id, feat_value = row['item.id'], row[feature]
    all_likes = xt_votes[((xt_votes['item.id'] == item_id) & (xt_votes['like'] == True))].shape[0]
    all_dislikes = xt_votes[((xt_votes['item.id'] == item_id) & (xt_votes['like'] == False))].shape[0]
    feat_likes = xt_votes[((xt_votes['item.id'] == item_id) & (xt_votes[feature] == feat_value) & (xt_votes['like'] == True))].shape[0]
    feat_dislikes = xt_votes[((xt_votes['item.id'] == item_id) & (xt_votes[feature] == feat_value) & (xt_votes['like'] == False))].shape[0]
    return feat_likes, feat_dislikes, all_likes, all_dislikes

def laplace_correction(t, b):
    if t==0 or b==0:
        return (t+1)/(b+2)
    return t/b

def ls_dataframe(xt_votes, users, feature):

    feat_votes = xt_votes[['item.id', feature]].drop_duplicates(ignore_index=True)
    fllist, fdlist, allist, adlist, odlist = [], [], [], [], []
    for i, row in feat_votes.iterrows():
        feat_likes, feat_dislikes, all_likes, all_dislikes = count_likes(row, feature, xt_votes)
        od = laplace_correction(laplace_correction(feat_likes, all_likes),laplace_correction(feat_dislikes,all_dislikes))
        fllist.append(feat_likes)
        fdlist.append(feat_dislikes)
        allist.append(all_likes)
        adlist.append(all_dislikes)
        odlist.append(od)
    feat_votes['like.feat'] = fllist
    feat_votes['dislike.feat'] = fdlist
    feat_votes['like.all'] = allist
    feat_votes['dislike.all'] = adlist
    feat_votes['odd.ratio'] = odlist

    return feat_votes

def generate_bayes_ls(votes, users, bayes_path = 'data/bayes/', verbose=True, save_to_csv = True):

    if not os.path.exists('data/bayes'):
            os.makedirs('data/bayes')

    xt_votes = extended_votes(votes, users)

    features = ['job', 'gender', 'age_group']
    feat_votes = []

    for feat in features:
        if verbose : print('Computing '+feat+' data')
        df_feat = ls_dataframe(xt_votes, users, feat)
        if save_to_csv:
            path = os.path.join(bayes_path, (feat+'.csv'))
            df_feat.to_csv(path, index=False, sep='|', mode='w')
        feat_votes.append(df_feat)

    if verbose : print('Dataframes generated')
    return feat_votes

# For fast execution, data are saved in csv files. switch read_from_csv to False to compute them
def gen_dataframes(votes, users, read_from_csv=True):
    if (read_from_csv):
        if not(os.path.isfile('data/bayes/job.csv') and os.path.isfile('data/bayes/gender.csv') and os.path.isfile('data/bayes/age_group.csv')):
            generate_bayes_ls(votes, users, verbose=False, save_to_csv=True)
        df_job = pd.read_csv('data/bayes/job.csv', sep="|")
        df_gender = pd.read_csv('data/bayes/gender.csv', sep="|")
        df_age_group = pd.read_csv('data/bayes/age_group.csv', sep="|")
    else:
        feat_votes = generate_bayes_ls(votes, users, verbose=False, save_to_csv=False)
        df_job = feat_votes[0]
        df_gender = feat_votes[1]
        df_age_group = feat_votes[2]
    return df_job, df_gender, df_age_group

def compute_odd_ratio(item_id, features, dfs):
    
    feat_job, feat_gender, feat_age_group = features
    df_job, df_gender, df_age_group = dfs

    row_job = df_job[(df_job['item.id'] == item_id) & (df_job['job'] == feat_job)]
    ls_job = 1.0
    if not(row_job.empty):
        ls_job = row_job.iloc[0]['odd.ratio']
    row_gender = df_gender[(df_gender['item.id'] == item_id) & (df_gender['gender'] == feat_gender)]
    ls_gender = 1.0
    if not(row_gender.empty):
        ls_gender = row_gender.iloc[0]['odd.ratio']
    row_age_group = df_age_group[(df_age_group['item.id'] == item_id) & (df_age_group['age_group'] == feat_age_group)]
    ls_age_group = 1.0
    if not(row_age_group.empty):
        ls_age_group = row_age_group.iloc[0]['odd.ratio']

    row = df_age_group[df_age_group['item.id'] == item_id].iloc[0]
    o_h = laplace_correction(row['like.all'],row['dislike.all'])

    odd = (o_h*ls_job*ls_gender*ls_age_group)
    return odd

def get_k_fav_item(items, features, dfs, k=10):
    topk = []
    for _, item in items.iterrows():
        odd = compute_odd_ratio(item['movie id '], features, dfs)
        heapq.heappush(topk, (odd, item))
        topk = heapq.nlargest(k, topk)
    return topk

#EndRegion

#Region[Red]

def compute_average_votes(model):
    like_ratings = []
    for item in model.T:
        likes = np.where(item>3, item, np.nan)
        if np.isnan(likes).all():
            mr_likes = 0
        else:
            mr_likes = np.nanmean(likes)
        dislikes = np.where(((item > 0) & (item <= 3)), item, np.nan)
        if np.isnan(dislikes).all():
            mr_dislikes = 0
        else:
            mr_dislikes = np.nanmean(dislikes)
        like_ratings.append((mr_likes, mr_dislikes))
    return like_ratings

def compute_error(model, votes, users):

    errors = []
    avg_votes = compute_average_votes(model)
    dfs = gen_dataframes(votes, users, read_from_csv=True)

    for _, vote in votes.iterrows():
        uid, iid = vote['user.id'], vote['item.id']
        features = get_features(users[users['id '] == uid].iloc[0])
        odd = compute_odd_ratio(iid, features, dfs)
        p = odd/(1+odd)
        pred_rating = p*avg_votes[iid][0] + (1-p)*avg_votes[iid][1]
        error = (vote['rating']-pred_rating)**2
        errors.append(error)

    mse = np.mean(errors)
    return mse

#EndRegion

#Region[Magenta] Run

print(compute_error(model, votes, users))

#EndRegion