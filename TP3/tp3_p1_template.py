
# -*- coding: utf-8 -*-
"""
TP2 Template-Problème 1
LOG6308
Auteur: Mikaël Perreault
Date: 5 janvier 2021
"""


from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs
import pprint
import matplotlib.pyplot as plt

from math import log, ceil
      
# Charger les données pour les votes
votes = tfds.load("movielens/100k-ratings", split="train")
# Charger les données pour les films
films = tfds.load("movielens/100k-movies", split="train")   
      
      
# On doit faire un mapping des attibuts nécessaires
#Pour votes, gardez les attributs "movie_title" et "user_id"
#Pour films, gardez les attributs "movie_title     

votes = votes.map(lambda x: {"movie_title": x["movie_title"], "user_id": x["user_id"]})

films = films.map(lambda x: x["movie_title"])

#Visualisez les données
for x in votes.take(10).as_numpy_iterator():
    pprint.pprint(x)

# On sépare notre ensemble de données en entraînement/test
# Affecter un seed=73 pour la constance des résultats et permtuez les données pour ne pas garder d'ordre particulier
seed = 73
tf.random.set_seed(seed)
shuffled = votes.shuffle(len(votes), seed=seed, reshuffle_each_iteration=False)

# Gardez 75% des données pour l'entraînement et 25% pour le test
train_size = int(len(shuffled) * 0.75)
train = shuffled.take(train_size) # 75000
test = shuffled.skip(train_size).take(len(shuffled)-train_size) # 25000

# On vérifie combien d'utilisateurs et de films uniques     
films_titres = films.batch(len(films))
user_ids = votes.batch(len(votes)).map(lambda x: x["user_id"])

# Question 2 : Combien y a-t-il d'utilisateurs uniques et de films uniques ?
unique_films_titres = np.unique(np.concatenate(list(films_titres)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

print("Question 2 :")
print("Nombre de films uniques "+str(len(unique_films_titres)))
print("Nombre d'utilisateurs uniques "+str(len(unique_user_ids)))
print("\n= = =\n")


# Question 3: Quels sont les avantages et inconvénients d'avoir un nombre de dimension élevé des représentations ?
# Plus le nombre de dimensions est élevé, plus le plongement permettra d'extraire des features précises et de dresser
# un profil de l'utilisateur et de l'objet : un plongement avec un grand nombre de dimensions permettra ainsi
# d'obtenir un résultat d'autant plus pertinent. Cependant, l'intérêt d'un plongement est de simplifier les entrées
# du réseau de neurone et de rendre l'apprentissage machine plus facile. Un plongement avec un nombre de dimensions
# trop important perd tout cet intérêt en ne simplifiant pas l'entrée, voire en la complexifiant.

embedding_dimension = 32


#On définit l'embedding côté utilisateur, on doit transformer les id User en représentation vectorielle
user_model = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=unique_user_ids, mask_token=None),
                                  tf.keras.layers.Embedding(len(unique_user_ids) + 1,
                                                            embedding_dimension)])
                                                            

# On définit maintenant l'embedding de la la portion film
film_model = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=unique_films_titres, mask_token=None),
                                   tf.keras.layers.Embedding(len(unique_films_titres) + 1,
                                                             embedding_dimension)])                              
                                                           
#On définit la métriques recherchées : FactorizedTopK
metrics = tfrs.metrics.FactorizedTopK(candidates=films.batch(128).map(film_model))

#On définit la tâche Retrieval en fonction de la métrique FactorizedTopK. 
task = tfrs.tasks.Retrieval(metrics=metrics)

class MovieLensModel(tfrs.Model):

  def __init__(self, user_model, film_model):
    super().__init__()
    self.film_model: tf.keras.Model = film_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    user_embeddings = self.user_model(features["user_id"])

    positive_film_embeddings = self.film_model(features["movie_title"])

    # La task calcule les métriques et le loss
    return self.task(user_embeddings, positive_film_embeddings)

#Question 4 : Donnez une piste de solution pour complexifier le présent modèle et tirer pleinement profit de l'approche réseau de neuronnes? Quelle serait sa principale limite?
#Réponse: Une piste pour complexifier cette approche serait d'utiliser un réseau plus complexe, constitué de plusieurs couche, avec par exemple une descente de gradients, afin de mettre à jour les poids et de
#         permettre au réseau d'apprendre et de s'actualiser en fonction de ses faiblesses au gré de l'apprentissage.
    
#Question 5 : Est-ce que le présent réseau de neuronne est plus, moins ou également performant par rapport à une approche de factorisation matricielle classique après 1 epoch d'entraînement?
#Réponse: après une seule epoch d'entrainement, le réseau de neurones peut être simplement ramené à une approche matricielle, puisque l'on va simplement multipliser les inputs par des
#         coefficients correspondant aux différents neurones. La performance sera donc similaire à une approche de factorisation matricielle classique.

model = MovieLensModel(user_model, film_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))


#Question 6 : Segmenter les batchs de manière à ce que le modèle roule 10 batch d'entraînement et 13 batchs de test par epoch, tout en ayant un batch size qui est de puissance n de 2.
train_batch_size = 2**ceil(log(len(train)/10,2))
test_batch_size = 2**ceil(log(len(test)/13,2))
cached_train = train.shuffle(len(train)).batch(train_batch_size).cache()
cached_test = test.batch(test_batch_size).cache()

#Question 7 : Entraînez le modèle avec model.fit jusqu’à ce qu’il surpasse de 40% la métrique top_100_categorical_top_k pour l’entraînement. Après combien d’epochs cela survient-t-il?
#Réponse: Le modèle dépasse 40% pour la métrique top_100_categorical_top_k à partir de 10 epochs.

top_100_categorical_top_k = 0
epoch = 0
loss = []
train_top_100 = []
test_top_100 = []
while top_100_categorical_top_k < 0.4:
    result = model.fit(cached_train, epochs=1)
    epoch += 1
    top_100_categorical_top_k = result.history['factorized_top_k/top_100_categorical_accuracy'][0]
    train_top_100.append(top_100_categorical_top_k)
    loss.append(result.history['total_loss'][0])

    test_top_100.append(model.evaluate(cached_test, return_dict=True)['factorized_top_k/top_100_categorical_accuracy'])


#Question 8 : Si on obtenait des données d'entraînement supplémentaires dans le futur, devrait-on recommencer l'entraînement du modèle sur l'ensemble du corpus augmenté des nouvelles données ? 
#Si oui, pourquoi ? Si non, comment devrait-on s'y prendre ? 
#Réponse: Non, il n'y a pas besoin de reprendre l'entraînement à 0. On peut séparer les nouvelles données en données de test et d'entrainement,
#         pour ensuite continuer l'entrainement de notre modèle. Il s'agit au final de faire une entrainement sur un modèle pré-entrainé.
#         Il faut cependant faire attention à ne pas tomber dans le surapprentissage qui peut survenir peu importe la méthode, et à veiller à ne pas
#         exécuter trop d'epochs d'apprentissage.

#Question 9 : Tracez le graphique du loss total d'entraînement en fonction des epochs
plt.plot(loss, marker='o')
plt.xticks(range(epoch), np.arange(1, epoch+1, 1))
plt.title("Loss total en fonction des epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss total")
plt.grid(color='0.95')
plt.show()

#Question 10 : Tracez, sur un même graphique, l'évolution de la métrique top_100_categorical_top_k en fonction des epochs pour l'entraînement et le test.

plt.plot(train_top_100, marker='o', label="entrainement")
plt.plot(test_top_100, marker='s', label="test")
plt.xticks(range(epoch), np.arange(1, epoch+1, 1))
plt.title("Evolution de top_100_categorical_top_k en fonction des epochs")
plt.xlabel("Epoch")
plt.ylabel("top_100_categorical_top_k")
plt.grid(color='0.95')
plt.legend(loc='upper left')
plt.show()

#Question 11 : Pourquoi la performance sur les données de test est inférieure à celle sur les données d'entraînement (1 raison)?
#Réponse:      La performance sur les données de test est plus faible par rapport aux données d'entrainement en raison d'un
#              surapprentissage : le modèle est "trop entraîné" sur les données de test et donc peu performant sur les données
#              qui n'appartiennent pas à cet ensemble de test.


#Question 12 : Est-ce que ces courbes représentent un résultat attendu pour un réseau de neuronne classique ? Pourquoi ? Développez. 
#Réponse:      Les courbes présentent un résultat innatendu par rapport à un réseau de neurone classique : en effet, bien que l'on
#              observe une diminution du loss lors de l'entrainement, et donc une amélioration de la performance au fur et à mesure
#              que l'on entraine le réseau de neurones, la performance sur les données de test dès la première epoch diminue ;
#              on pourrait penser que le réseau est dès la première itération d'entrainement en surapprentissage.

#Question 13 : Recommandez les 5 meilleurs films pour l'utilisateur 25

index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)

index.index(films.batch(100).map(model.film_model), films)


print("\nQuestion 13:")
_, titles = index(tf.constant(["25"]))
print("Recommendations for user 42:")
for title in titles[0, :5]:
    print("| "+title.numpy().decode("utf-8"))
