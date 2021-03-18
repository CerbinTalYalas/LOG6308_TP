import pandas as pd
import numpy as np

<<<<<<< Updated upstream
adjacent = pd.read_table("../data/citeseer.rtable", sep=" ")
=======
>>>>>>> Stashed changes

def cos_matrix(A):
    # code from (https://stackoverflow.com/a/20687984)
    # base similarity matrix (all dot products)
    # replace this with A.dot(A.T).toarray() for sparse representation
    similarity = np.dot(A, A.T)

    # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    # inv_mag represent inv magnitude for each vector
    cosine = similarity * inv_mag  # multiply value  by the first inv mag
    cosine = cosine.T * inv_mag  # multiply value by the second inv mag
    return cosine


def compute_cos_similarity(adjacent, top_n=10):
    cos = cos_matrix(adjacent.values)
    # remove diagonal to don't recommend it self
    np.fill_diagonal(cos, 0)
    result = pd.DataFrame(adjacent.index.to_numpy()[cos.argsort(1)[:, :-(top_n+1):-1]], index=adjacent.index)
    result = result.rename(columns=lambda x: 'Top_{}'.format(x + 1))
    return result


def get_doc_recommendation(cos_similarity_result, id):
    return cos_similarity_result.loc[id]
