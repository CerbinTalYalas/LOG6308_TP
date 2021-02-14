import numpy as np


def cos_matrix(model):
    cos_matrix = 1.0 * np.matmul(model.transpose(), model)
    nb_items = len(model[0])
    for i in range(nb_items):
        norm = np.linalg.norm(model[:, i])
        cos_matrix[i] = cos_matrix[i] / norm
        cos_matrix[:, i] = cos_matrix[:, i] / norm
        cos_matrix[i, i] = 1.0 * np.NINF
    return cos_matrix


def closest_neighbor(dist_items, k):
    return np.argpartition(dist_items, -k, axis=0)[-k:]


def compute_K(v_iid, w_iid):
    w_i = np.zeros(v_iid.shape)
    for u, v_ui in enumerate(v_iid):
        c_ui = np.ma.masked_where(v_ui != 0, v_ui).mask
        w_i[u] = c_ui * w_iid
    w_i = w_i.sum(axis=1)
    w_i = np.true_divide(1, w_i, out=np.zeros_like(w_i), where=w_i!=0)
    return w_i


def predict_i(b, K, v_0, w):
    return (b + K * np.matmul(v_0, w)).clip(1,5)


def compute_mean_R_0(R):
    mask = np.ma.masked_where(R == 0, R)
    mean = mask.mean(axis=0).filled(0)
    return mean, (mask - mean).filled(0)


def mse(votes_pd, votes_predicted):
    err = np.full(votes_predicted.shape, np.nan)
    for _, vote in votes_pd.iterrows():
        pred = votes_predicted[vote["user.id"], vote["item.id"]]
        dif = (vote["rating"] - pred)
        err[vote["user.id"], vote["item.id"]] = (dif ** 2)
    return np.nanmean(err, axis=0)


def votes_communs(model):
    model_vote = np.ma.masked_where(model != 0, model).mask
    model_vote = model_vote.astype(int)
    nb_vote_commun = np.matmul(model_vote.transpose(), model_vote)
    np.fill_diagonal(nb_vote_commun, 0)
    return nb_vote_commun

def nb_voisins(data):
    voisin = np.ma.masked_where(data != 0, data).mask
    return voisin.sum(axis=0) #2b

def compute_error(model, ind, w, votes):
    
    mean, R_0 = compute_mean_R_0(model)
    res = np.zeros(model.shape)
    for iid in range(model.shape[1]):
        ind_i = ind[:, iid]
        w_i = w[:, iid]
        v_0_i = R_0[:, ind_i]
        K_i = compute_K(v_0_i,w_i)
        res[:,iid] = predict_i(mean[iid], K_i, v_0_i, w_i)

    err = mse(votes, res)
    mean_mse = np.nanmean(err)

    return err, mean_mse