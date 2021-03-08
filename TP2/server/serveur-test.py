# -*- coding: utf-8 -*-

import requests

r=requests.get('http://localhost:8080/422908')

if r.text != 200:
    print(r.text)
else:
    print("La requête au serveur a échoué")
    
