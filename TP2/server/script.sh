# Importation des données décompressées dans Mongo
# mongoimport --db test --collection oai --file oai.mongo

# Démarrage du serveur qui écoute sur le port 8080
python serveur-oai.py

# Affichage des données de l'article 422908 dans un fureteur à l'adresse http://localhost:8080/422908
