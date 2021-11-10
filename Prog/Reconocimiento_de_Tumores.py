#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 21:17:03 2020 

@author: Franciso & Gustavo
"""
# Importamos las bibliotecas
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron 
from imgUsuario import *




#Separamos las clase de las etiquetas
# Clase
X = datosFull.iloc[:, 0:-1]
# Etiquetas
Y = datosFull.iloc[:, -1]

# Divide matrices en sub conjuntos de pruebas y trenes aleatorio
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.3, random_state=0)

# Mandamos a llamar al perceptron para trabajar con él
perceptron = Perceptron()

# Se usa la función fit para entrenar al perceptron con los datos ya dados
perceptron.fit(X_train, y_train)

# Eficiencia del algoritmo 
print(perceptron.score(X_test, y_test))

# Imprimimos si el valor es -1 0 1 dependiendo la salida
print("La predicción es: ", int(perceptron.predict(imT))) 

# Para que el usuario entienda mas imprimimos la respuesta si es o no humano
if((int(perceptron.predict(imT))) == 1):
    print("Es un tumor =c")
else:
    print("No es tumor ")