import numpy as np

#Region[Red] SVD Decomposition

def svd_decomp(model):
    u,s,vh = np.linalg.svd(model, full_matrices = False)
    return u, s, vh

def svd_decomp_reduc(model, k):
    u, s, vh = svd_decomp(model)
    s = s * (k*[1.0]+(len(s)-k)*[0.0])
    return u, np.diag(s), vh

#EndRegion

#Region[Cyan] Prediction

def square_error(model, k=10):
    u, s, vh = svd_decomp_reduc(model, k)
    svd_matrix = u @ s @ vh
    err = (model-svd_matrix)
    print(err)
    return np.linalg.norm(err, ord='fro')/np.count_nonzero(err)

#EndRegion