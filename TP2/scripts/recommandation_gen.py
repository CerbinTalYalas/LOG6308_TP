import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.cos_similarity import compute_cos_similarity
from scripts.pagerank import compute_pagerank, get_top_pagerank


def compute_tfidf_recommandation(abstract, top_n=20):
    abstract_copy = abstract.copy().dropna()
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', use_idf=True, smooth_idf=True)
    tfidf_result = vectorizer.fit_transform(abstract_copy['Description']).toarray()

    pd_tfidf = pd.DataFrame(tfidf_result, index= abstract_copy['Id'].tolist())

    cos_res = compute_cos_similarity(pd_tfidf, top_n=top_n)
    return cos_res


def compute_cos_recommandation(adjacent, top_n=20):
    cos_res = compute_cos_similarity(adjacent, top_n=top_n)
    return cos_res


def dense_to_full(dataframe):
    result = pd.DataFrame(0, index=dataframe.index, columns=dataframe.index)
    for index, row in dataframe.iterrows():
        #recommandation_row = recommandation.loc[index]
        for val in row.values[~pd.isnull(row.values)]:
            result.loc[index, val] += 1

    return result

def gen_recommandation(adjacent, abstract):
    recommandation_tfidf = compute_tfidf_recommandation(abstract, top_n= 20).astype('Int64')
    recommandation_cos = compute_cos_similarity(adjacent, top_n=20)

    joined_recomandation = recommandation_cos.join(recommandation_tfidf, lsuffix="_cos", rsuffix="_tfidf")

    full_joined_recommandation = dense_to_full(joined_recomandation)

    pagerank = compute_pagerank(adjacent)
    dense_result = get_top_pagerank(full_joined_recommandation, pagerank, top_n=20)

    full_result = dense_to_full(dense_result)

    return full_result