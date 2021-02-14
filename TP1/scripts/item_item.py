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