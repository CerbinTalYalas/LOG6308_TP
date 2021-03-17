import pandas as pd
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer

import scripts.cos_similarity

abstract = pd.read_table("../data/abstracts.csv", sep=",")
abstract.drop(abstract.columns[0], axis=1, inplace=True)
abstract = abstract.dropna()

adjacent = pd.read_table("../data/citeseer.rtable", sep=" ")

# remove punctuation
regex = re.compile(r'[^\w\s]+')
# abstract['tokenized_Titre'] = [regex.sub('', x) for x in abstract['Titre'].tolist()]
abstract['tokenized_Description'] = [regex.sub('', x) for x in abstract['Description'].tolist()]

# lower case
# abstract['tokenized_Titre'] = abstract['tokenized_Titre'].str.lower()
abstract['tokenized_Description'] = abstract['tokenized_Description'].str.lower()

# tokenize
# abstract['tokenized_Titre'] = abstract["tokenized_Titre"].apply(nltk.word_tokenize)
abstract['tokenized_Description'] = abstract["tokenized_Description"].apply(nltk.word_tokenize)

# remove stop word
stop_word = set(nltk.corpus.stopwords.words('english'))
# abstract['tokenized_Titre'] = abstract['tokenized_Titre'].apply(lambda x: [word for word in x if word not in stop_word])
abstract['tokenized_Description'] = abstract['tokenized_Description'].apply(lambda x: [word for word in x if word not in stop_word])

# stemming of word
stemmer = nltk.stem.snowball.EnglishStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()
# abstract['stemmed_Titre'] = abstract['tokenized_Titre'].apply(lambda x: [stemmer.stem(word) for word in x])
# abstract['stemmed_Description'] = abstract['tokenized_Description'].apply(lambda x: [stemmer.stem(word) for word in x])
abstract['stemmed_Description'] = abstract['tokenized_Description'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# get most frequent word
# data = abstract['stemmed_Titre'].explode().dropna().astype('string')
# word_counter_titre = nltk.Counter(data)
# most_common_titre = [word[0] for word in word_counter_titre.most_common(20)]
data = abstract['stemmed_Description'].explode().dropna().astype('string')
word_counter_desc = nltk.Counter(data)
# most_common_desc = [word[0] for word in word_counter_desc.most_common(20)]

# remove most frequent word
# abstract['stemmed_Titre'] = abstract['stemmed_Titre'].apply(lambda x: [word for word in x if word not in most_common_titre])
# abstract['stemmed_Description'] = abstract['stemmed_Description'].apply(lambda x: [word for word in x if word not in most_common_desc])

# abstract['stemmed_Description'] = abstract['stemmed_Description'].apply(lambda x: ' '.join(x))

vectorizer = TfidfVectorizer(analyzer= lambda x: x)
#response1 = vectorizer.fit_transform(abstract['stemmed_Titre'])
#response1 = response1.toarray()

response2 = vectorizer.fit_transform(abstract['stemmed_Description'])
response2 = response2.toarray()

#adjacent_titre = pd.DataFrame(response1, index= abstract['Id'].tolist())
adjacent_desc = pd.DataFrame(response2, index= abstract['Id'].tolist())

#cos_titre = scripts.cos_similarity.compute_cos_similarity(adjacent_titre, 20)
#res1_422908 = scripts.cos_similarity.get_doc_recommendation(cos_titre, 422908)
cos_desc = scripts.cos_similarity.compute_cos_similarity(adjacent_desc, 20)
res2_422908 = scripts.cos_similarity.get_doc_recommendation(cos_desc, 422908)

vectorizer = TfidfVectorizer(stop_words='english')
response3 = vectorizer.fit_transform(abstract['Description'])
response3 = response3.toarray()

adjacent_no_treat = pd.DataFrame(response3, index= abstract['Id'].tolist())
cos_no_treat = scripts.cos_similarity.compute_cos_similarity(adjacent_no_treat, 20)
res3_422908 = scripts.cos_similarity.get_doc_recommendation(cos_no_treat, 422908)


to_find = np.ma.masked_where(adjacent.loc[422908].to_numpy() == 1, adjacent.index.to_numpy())
to_find = adjacent.index.to_numpy()[to_find.mask]


def check(array1, array2):
    count = 0
    for element in array1:
        if element in array2: count += 1

    return count

#element_found1 = check(res1_422908.to_numpy(), to_find)
#print("tfidf custom token titre : ", element_found1)
element_found2 = check(res2_422908.to_numpy(), to_find)
print("tfidf custom token description : ", element_found2)
element_found3 = check(res3_422908.to_numpy(), to_find)
print("tfidf basic description : ", element_found3)