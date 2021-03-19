import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts import cos_similarity


def compute_tfidf_recommandation(abstract, top_n=10):

    # tokenize word
    abstract['tokenized_Description'] = abstract["Description"].apply(nltk.word_tokenize)

    # lemmatisation
    lemmatizer = nltk.stem.WordNetLemmatizer()
    abstract['lemmed_Description'] = abstract['tokenized_Description'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    # because we found no improvement for deeper manual preprocessing we prefer to use TfidfVectorizer default one
    # put token back to sentance for TFIDF
    abstract['lemmed_Description'] = abstract['lemmed_Description'].apply(lambda x: ' '.join(x))


    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', use_idf=True, smooth_idf=True)
    tfidf_result = vectorizer.fit_transform(abstract['lemmed_Description']).toarray()

    pd_tfidf = pd.DataFrame(tfidf_result, index= abstract['Id'].tolist())

    cos_res = cos_similarity.compute_cos_similarity(pd_tfidf, top_n=top_n)
    return cos_res
