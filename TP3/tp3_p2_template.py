# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 01:00:42 2021

@author: mikap
"""


# -*- coding: utf-8 -*-
"""
TP2 corrigé-Problème 2
LOG6308
Auteur: Mikaël Perreault
Date: 5 janvier 2021
"""
#import os
#import pprint
#import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pprint
import tensorflow_recommenders as tfrs
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from math import log, ceil
pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns',30)
pd.set_option('display.max_rows',100)
pd.set_option('display.max_colwidth',100)
pd.set_option('display.max_rows',100)
pd.set_option('display.width',None)
# Charger les données pour les votes
votes = tfds.load("movielens/100k-ratings", split="train")
# Charger les données pour les films
films = tfds.load("movielens/100k-movies", split="train")   
      
      
# On doit faire un mapping des attibuts nécessaires
#Pour votes, gardez les attributs "movie_title", "user_id" et "user_rating"
#Pour films, gardez les attributs "movie_title"
votes = votes.map(lambda x: {"movie_title": x["movie_title"],"user_id": x["user_id"],"user_rating": x["user_rating"]})
sample = votes.map(lambda x: {"movie_title": x["movie_title"],"user_id": x["user_id"]})#avec que movie title et user id
films = films.map(lambda x: x["movie_title"])

# On sépare notre ensemble de données en entraînement/test
# Affecter un seed=73 pour la constance des résultats et permtuez les données pour ne pas garder d'ordre particulier
tf.random.set_seed(73)
shuffled = votes.shuffle(len(votes), seed=73, reshuffle_each_iteration=False)

#Gardez 75% des données pour l'entraînement et 25% pour le test
train_size = int(len(shuffled) * 0.75)
train = shuffled.take(train_size) # 75000
test = shuffled.skip(train_size).take(len(shuffled)-train_size)

# On vérifie combien d'utilisateurs et de films uniques     
movie_title = films.batch(len(films))
user_ids = votes.batch(len(votes)).map(lambda x: x["user_id"])

unique_movie_title = np.unique(np.concatenate(list(movie_title)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

class RankingModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    embedding_dimension = 32

    # Embeddings user
    self.user_embeddings = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)])

    # Embeddings film.
    self.film_embeddings = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_movie_title, mask_token=None),
      tf.keras.layers.Embedding(len(unique_movie_title) + 1, embedding_dimension)])

    #Calcule les prédictions
    self.votes = tf.keras.Sequential([
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      tf.keras.layers.Dense(1)
      ])

  def call(self, inputs):

    user_id, movie_title = inputs

    user_embedding = self.user_embeddings(user_id)
    film_embedding = self.film_embeddings(movie_title)
    return self.votes(tf.concat([user_embedding, film_embedding], axis=1)), film_embedding

#Il s'agit d'une tâche qui prend les valeurs vraies et prédites et qui retournent le loss ainsi que le RMSE associé 
task = tfrs.tasks.Ranking(loss = tf.keras.losses.MeanSquaredError(),metrics=[tf.keras.metrics.RootMeanSquaredError()])


class MovieLensModel(tfrs.models.Model):

  def __init__(self):
    super().__init__()
    self.ranking_model: tf.keras.Model = RankingModel()
    
    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
      loss = tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

  def call(self, features):
    votes_predictions, film_embedding = self.ranking_model(
        (features["user_id"], features["movie_title"]))
    return votes_predictions, film_embedding


  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    votes_predictions, film_embedding = self.ranking_model(
        (features["user_id"], features["movie_title"]))

    
    return self.task(labels=features["user_rating"], predictions=votes_predictions)


model = MovieLensModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

#Segmenter les batchs de manière à ce que le modèle roule 10 batch d'entraînement et 13 batchs de test par epoch, tout en ayant un batch size qui est un multiple de 2^n.  
train_batch_size = 2**ceil(log(len(train)/10,2))
test_batch_size = 2**ceil(log(len(test)/13,2))
cached_train = train.shuffle(len(train)).batch(train_batch_size).cache()
cached_test = test.batch(test_batch_size).cache()
cached_sample = sample.batch(len(sample))

#Question 3 : Entraînez le modèle jusqu'à ce que le RMSE de l'entraînement soit inférieur à 0.925
RMSE = 1
epoch = 0
RMSE_Train = []
RMSE_Test = []
while RMSE > 0.925:
    result = model.fit(cached_train, epochs=1)
    RMSE = result.history['root_mean_squared_error'][0]
    RMSE_Train.append(RMSE)
    RMSE_Test.append(model.evaluate(cached_test, return_dict=True)['root_mean_squared_error'])
    epoch += 1

print("Question 3 : RMSE inférieur à 0.925 au bout de ", epoch, " epochs")

#Question 4 : Faites le graphique du RMSE d'entraînement et de test selon le nombre d'epochs. Pour combien d'epochs devraient-on entraîner ce modèle?
#Réponse: On devrait entraîner ce modèle pour 20 epochs. A partir de 20 epochs, la RMSE ne diminue plus.

data = pd.DataFrame(dict(Epoch=np.arange(1, epoch+1, 1), Train = RMSE_Train, Test = RMSE_Test))
data = pd.melt(data, ["Epoch"], var_name="type", value_name="RMSE")
sns.set_theme()
g = sns.lineplot(
  data = data, x="Epoch", y="RMSE", hue="type"
)
g.set_title("Evolution des RMSE en fonction des epochs")
g.set_xticks(np.arange(1, epoch+1, 2))
g.set_xticklabels(np.arange(1, epoch+1, 2))
plt.show()
"""
plt.plot(RMSE_Train, label="entrainement")
plt.plot(RMSE_Test, label="test")
plt.xticks(range(epoch), np.arange(1, epoch+1, 2))
plt.title("Evolution des RMSE en fonction des epochs")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.grid(color='0.95')
plt.legend(loc='upper left')
plt.show()
"""

# Question 5 : Vous devez implémenter la méthode call dans MovieLensModel pour exécuter la prochaine ligne.
prediction, embedding = model.predict(cached_sample)


#Question 6 : Vous devez reprédire les votes avec le modèle entraîné. On va
#considérer 4 modèles : 1) après 1 epoch, 2) après 5 epochs 3) après 10 epochs et 4) après 20 epochs
#Vous devez produire, pour chaque modèle un boxplot des prédictions pour chaque vote (ex: répartition des prédictions pour un rating attendu de 1, de 2... jusqu'à 5)
#Commentez vos observations sur l'évolution du graphique en fonction du nombre d'epochs.
model_1_epoch = MovieLensModel()
model_5_epoch = MovieLensModel()
model_10_epoch = MovieLensModel()
model_20_epoch = MovieLensModel()

model_1_epoch.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
model_5_epoch.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
model_10_epoch.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
model_20_epoch.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

model_1_epoch.fit(cached_train, epochs=1)
model_5_epoch.fit(cached_train, epochs=5)
model_10_epoch.fit(cached_train, epochs=10)
model_20_epoch.fit(cached_train, epochs=20)

prediction_1_epoch, embedding_1_epoch = model_1_epoch.predict(cached_sample)
prediction_5_epoch, embedding_5_epoch = model_5_epoch.predict(cached_sample)
prediction_10_epoch, embedding_10_epoch = model_10_epoch.predict(cached_sample)
prediction_20_epoch, embedding_20_epoch = model_20_epoch.predict(cached_sample)

votes_value = votes.map(lambda x: x["user_rating"])
votes_list = list(tfds.as_numpy(votes_value))

df_votes = pd.DataFrame(dict(Epoch_1 = prediction_1_epoch.flatten(),
                              Epoch_5 = prediction_5_epoch.flatten(),
                              Epoch_10 = prediction_10_epoch.flatten(),
                              Epoch_20 = prediction_20_epoch.flatten(),
                              Expected_vote = votes_list))


fig, axes = plt.subplots(nrows= 2, ncols = 2, figsize=(10,10))
fig.suptitle("Comparaison des performances selon le nombre d'epochs d'entrainement")
axes[0,0].set(ylim=(1,5))
axes[0,1].set(ylim=(1,5))
axes[1,0].set(ylim=(1,5))
axes[1,1].set(ylim=(1,5))

sns.boxplot(ax = axes[0,0], x="Expected_vote", y="Epoch_1", data=df_votes, showfliers=False)
sns.boxplot(ax = axes[0,1], x="Expected_vote", y="Epoch_5", data=df_votes, showfliers=False)
sns.boxplot(ax = axes[1,0], x="Expected_vote", y="Epoch_10", data=df_votes, showfliers=False)
sns.boxplot(ax = axes[1,1], x="Expected_vote", y="Epoch_20", data=df_votes, showfliers=False)
#ax.set(ylim=(1,5))
plt.show()

#Réponse: On observe que pour un petit nombre d'epoch, la prédiction varie peu : en particulier, pour une seule epochs autour de 3 (le vote moyen standard,
#         comme les votes varient entre 1 et 5), et pour 5 epochs autour de 3.5 (la moyenne sur l'ensemble d'entraînement). Cependant, à partir de 5 epochs,
#         on remarque que les prédictions commencent à se différencier selon le vote attendu, et à s'ordonner : plus le vote attendu est grand, plus le vote
#         prédit sera plus grand que le vote moyen, et inversement. En d'autres termes, on constate que les votes prédits se différencient et se précisent pour
#         correspondre aux votes attendus. Cependant, il faut garder en tête qu'un trop grand nombre d'epochs pourrait mener à du surapprentissage.


# 7 Compilez, dans un dataframe Pandas, les moyennes ainsi que les ecart-types des predictions pour chaque modèle et chaque vote.
#Les lignes du dataframe devraient être : [1ep,5ep,10ep,20ep] et les colonnes devraient être : [moy_1,moy_2,moy_3,moy_4,moy_5,ec_1,ec_2,ec_3,ec_4,ec_5]
df_votes = pd.melt(df_votes, ["Expected_vote"], var_name="Epoch", value_name="Predicted_vote")
df_grouped_votes = df_votes.groupby(["Epoch", "Expected_vote"]).agg({'Predicted_vote': ['mean', 'std']})

df_result = pd.DataFrame(index=["1ep","5ep","10ep","20ep"],
                         columns=["moy_1","moy_2","moy_3","moy_4","moy_5","ec_1","ec_2","ec_3","ec_4","ec_5"])

df_result.loc["1ep"] = df_grouped_votes.loc["Epoch_1"].to_numpy().transpose().flatten()
df_result.loc["5ep"] = df_grouped_votes.loc["Epoch_5"].to_numpy().transpose().flatten()
df_result.loc["10ep"] = df_grouped_votes.loc["Epoch_10"].to_numpy().transpose().flatten()
df_result.loc["20ep"] = df_grouped_votes.loc["Epoch_20"].to_numpy().transpose().flatten()

print("Question 7 :")
print("Moyennes et écart-types à différentes epochs\n")
print(df_result)
print("\n= = =")

#Question 8 : En vous servant de la distance cosinus, effectuez un calcul de similarité entre les embeddings des
#films pour retrouver les 5 films les plus semblables à 1) Pulp Fiction (id 3)
#2) Silence of the Lambs (id 43) et 3) 2001: A Space Odyssey (id 264). Affichez les résultats.


def top5_similar(iid, film_names):
    cos_similarity = 1 - scipy.spatial.distance.cdist([embedding[iid]], embedding, 'cosine')
    unique_cos, unique_index = np.unique(cos_similarity, return_index=True)

    top5_index = np.flip(unique_index[-6:-1])
    top5_cos = np.flip(unique_cos[-6:-1])

    for (index, cos) in zip(top5_index, top5_cos):
        print("| "+film_names[index].decode("utf-8")+" - "+str(cos)[0:4])

print("\nQuestion 8 :")
film_names = list(tfds.as_numpy(votes.map(lambda x: x["movie_title"])))
print("5 films les plus semblables à Pulp Fiction (id 3)")
top5_similar(3, film_names)
print("= = =")
print("5 films les plus semblables à Silence of the Lambs (id 43)")
top5_similar(43, film_names)
print("= = =")
print("5 films les plus semblables à 2001: A Space Odyssey (id 264)")
top5_similar(264, film_names)
print("= = =")