import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.cos_similarity import compute_cos_similarity, compute_cos_similarity_full
from scripts.pagerank import compute_pagerank, get_top_pagerank


def compute_tfidf_recommandation(abstract, index,top_n=20):
    abstract_copy = abstract.copy().dropna()
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', use_idf=True, smooth_idf=True)
    tfidf_result = vectorizer.fit_transform(abstract_copy['Description']).toarray()

    pd_tfidf = pd.DataFrame(tfidf_result, index= abstract_copy['Id'].tolist())

    cos_res = compute_cos_similarity(pd_tfidf, top_n=top_n)
    # allow result to be in the same shape as other dataframe
    cos_res = dense_to_full(cos_res, index)
    return cos_res


def dense_to_full(dataframe, index):
    result = pd.DataFrame(0, index=index, columns=index)
    for index, row in dataframe.iterrows():
        result.loc[index, row] = 1

    return result


def fn_recommandations(adjacent, abstract):

    recommandation_tfidf = compute_tfidf_recommandation(abstract, adjacent.index, top_n=20)
    recommandation_cos = compute_cos_similarity_full(adjacent, adjacent.index, top_n=20)
    full_recommandation = recommandation_cos.add(recommandation_tfidf, fill_value = 0)

    pagerank = compute_pagerank(adjacent)
    dense_result = get_top_pagerank(full_recommandation, pagerank, top_n=20)

    full_result = dense_to_full(dense_result, adjacent.index)

    return full_result