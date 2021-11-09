# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 21:17:03 2020 

@author: Franciso & Gustavo
"""
import cv2
import numpy as np
from PIL import Image as im
import cv2
from im import *




    #Ingresamos la imagen que servira como entrenamiento a la neurona (IMAGEN UNO H)
img = cv2.imread(rutaG, 0) #Leemos la imagen
    
    #Para redimensionar la imagen y hacer que sean menos los pieles del entrenamiento,
    #tilizaremos la opcion de opencv "resize" para definir la medida de la imagen
primerF = cv2.resize(img, (100, 300))#(ancho->columnas)(largo->Renglones)
cv2.imwrite(rutaR,primerF)

#Ahora ya aplicado estos filtros haremos la rotación de imagenes    

    #Rotacion 90 grados
# Primero obtendremos largo y ancho de la imagen
height, width = primerF.shape[0:2]
imR = cv2.getRotationMatrix2D((width/2, height/2),90,.30)
rot90 = cv2.warpAffine(primerF, imR, (width, height))
cv2.imwrite(rutaN,rot90)
    
    #Rotacion 180 grados
# Primero obtendremos largo y ancho de la imagen
height, width = primerF.shape[0:2]
imR = cv2.getRotationMatrix2D((width/2, height/2),180,1)
rot180 = cv2.warpAffine(primerF, imR, (width, height))
cv2.imwrite(rutaC,rot180)

    #Rotacion 270 grados
# Primero obtendremos largo y ancho de la imagen
height, width = primerF.shape[0:2]
imR = cv2.getRotationMatrix2D((width/2, height/2),270,.30)
rot270 = cv2.warpAffine(primerF, imR, (width, height))
cv2.imwrite(rutaD,rot270)
        
    #Rotacion 360 grados
# Primero obtendremos largo y ancho de la imagen
height, width = primerF.shape[0:2]
imR = cv2.getRotationMatrix2D((width/2, height/2),360,1)
rot360 = cv2.warpAffine(primerF, imR, (width, height))
cv2.imwrite(rutaT,rot360)
"""""




#Leemos las imagenes rotadas para despues convertirlas a una matriz
# Imagenes rotadas 1/Hombre
img0 = cv2.imread("/home/franck2407/Downloads/ProyectoFinal/ImagenesRotadasO/1/H/Imagen_90_G.jpg", 0) #Leemos la imagen a BN
img1 = cv2.imread("/home/franck2407/Downloads/ProyectoFinal/ImagenesRotadasO/1/H/Imagen_180_G.jpg", 0) #Leemos la imagen a BN
img2 = cv2.imread("/home/franck2407/Downloads/ProyectoFinal/ImagenesRotadasO/1/H/Imagen_270_G.jpg", 0) #Leemos la imagen a BN
img3 = cv2.imread("/home/franck2407/Downloads/ProyectoFinal/ImagenesRotadasO/1/H/Imagen_360_G.jpg", 0) #Leemos la imagen a BN

# Convertir las imagenes a un arreglo de numpy 

img00 = np.array(img0)
img11 = np.array(img1)
img22 = np.array(img2)
img33 = np.array(img3)


# APlanar el arreglo a un vector de 30,000 x 1 

img000 = img00.reshape([30000,1]) #Tamaño dinámico
img111 = img11.reshape([30000,1]) #Tamaño dinámico
img222 = img22.reshape([30000,1]) #Tamaño dinámico
img333 = img33.reshape([30000,1]) #Tamaño dinámico

# Dividir cada elemento entre 255 para mejor eficiencia

X0 = img000/255
X1 = img111/255
X2 = img222/255
X3 = img333/255









#Ruta de la imagen a tratar
global ruta1
ruta1 = ("C:/Users/franc/Documents/ClasificadorCancerCerebral/ImagenesOriginalesO/1.jpg")
#Ruta de la imagen reducida
global rutaR
rutaR = ("C:/Users/franc/Documents/ClasificadorCancerCerebral/ImagenesReducidasO/r1.jpg")
#Ruta de las imagenes guaradas rotadas
global ruta90
ruta90 = ("C:/Users/franc/Documents/ClasificadorCancerCerebral/ImagenesRotadasO/1_90.jpg")
global ruta180
ruta180 = ("C:/Users/franc/Documents/ClasificadorCancerCerebral/ImagenesRotadasO/1_180.jpg")
global ruta270
ruta270 = ("C:/Users/franc/Documents/ClasificadorCancerCerebral/ImagenesRotadasO/1_270.jpg")




"""



















