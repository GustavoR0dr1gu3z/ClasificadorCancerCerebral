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
# Leemos la imagen a predecir

#Ruta Franck
img = cv2.imread("C:/Users/franc/Documents/ClasificadorCancerCerebral/Imagenes_A_Predecir/2.jpg",0)
#Ruta Gustavo
img = cv2.imread("C:/Users/franc/Documents/ClasificadorCancerCerebral/Imagenes_A_Predecir/2.jpg",0)

# Reeducimos la imagen a 100x300 pixeles para pasar a un arreglo de numpy
primerF = cv2.resize(img, (100, 300))#(ancho->columnas)(largo->Renglones)

# Pasamos esta imagen ya redimencionada a un arreglo
img44 = np.array(primerF)

# Aplanamos el arreglo y lo hacemos de 30,000(Filas) x 1(Columna)
img444 = img44.reshape([30000,1]) #Tamaño dinámico

# Lo dividimos para optimizar el funcionamiento
xf = img444/255

