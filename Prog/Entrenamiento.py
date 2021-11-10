# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 21:17:03 2020 

@author: Franciso & Gustavo
"""
import cv2
import numpy as np
from PIL import Image as im
import cv2

    #Ingresamos la imagen que servira como entrenamiento a la neurona (IMAGEN UNO H)
img = cv2.imread("../ImagenesOriginalesO/1.jpg", 0) #Leemos la imagen

    #Para redimensionar la imagen y hacer que sean menos los pieles del entrenamiento,
    #tilizaremos la opcion de opencv "resize" para definir la medida de la imagen
primerF = cv2.resize(img, (100, 300))#(ancho->columnas)(largo->Renglones)
cv2.imwrite("../ImagenesReducidasO/R1.jpg",primerF)

#Ahora ya aplicado estos filtros haremos la rotación de imagenes    

    #Rotacion 90 grados
# Primero obtendremos largo y ancho de la imagen
height, width = primerF.shape[0:2]
imR = cv2.getRotationMatrix2D((width/2, height/2),90,.30)
rot90 = cv2.warpAffine(primerF, imR, (width, height))
cv2.imwrite("../ImagenesRotadasO/1_R_T.jpg",rot90)
    
    #Rotacion 180 grados
# Primero obtendremos largo y ancho de la imagen
height, width = primerF.shape[0:2]
imR = cv2.getRotationMatrix2D((width/2, height/2),180,1)
rot180 = cv2.warpAffine(primerF, imR, (width, height))
cv2.imwrite("../ImagenesRotadasO/1_R_D.jpg",rot180)

    #Rotacion 270 grados
# Primero obtendremos largo y ancho de la imagen
height, width = primerF.shape[0:2]
imR = cv2.getRotationMatrix2D((width/2, height/2),270,.30)
rot270 = cv2.warpAffine(primerF, imR, (width, height))
cv2.imwrite("../ImagenesRotadasO/1_R_C.jpg",rot270)
        
    #Rotacion 360 grados
# Primero obtendremos largo y ancho de la imagen
height, width = primerF.shape[0:2]
imR = cv2.getRotationMatrix2D((width/2, height/2),360,1)
rot360 = cv2.warpAffine(primerF, imR, (width, height))
cv2.imwrite("../ImagenesRotadasO/1_R_N.jpg",rot360)



# Imagenes rotadas
img0 = cv2.imread("../ImagenesRotadasO/100_R_T.jpg", 0) #Leemos la imagen a BN
img1 = cv2.imread("../ImagenesRotadasO/100_R_D.jpg", 0) #Leemos la imagen a BN
img2 = cv2.imread("../ImagenesRotadasO/100_R_C.jpg", 0) #Leemos la imagen a BN
img3 = cv2.imread("../ImagenesRotadasO/100_R_N.jpg", 0) #Leemos la imagen a BN
'''
img4 = cv2.imread("../ImagenesRotadasO/96_R_T.jpg", 0) #Leemos la imagen a BN
img5 = cv2.imread("../ImagenesRotadasO/96_R_D.jpg", 0) #Leemos la imagen a BN
img6 = cv2.imread("../ImagenesRotadasO/96_R_C.jpg", 0) #Leemos la imagen a BN
img7 = cv2.imread("../ImagenesRotadasO/96_R_N.jpg", 0) #Leemos la imagen a BN

img8 = cv2.imread("../ImagenesRotadasO/97_R_T.jpg", 0) #Leemos la imagen a BN
img9 = cv2.imread("../ImagenesRotadasO/97_R_D.jpg", 0) #Leemos la imagen a BN
img10 =cv2.imread("../ImagenesRotadasO/97_R_C.jpg", 0) #Leemos la imagen a BN
img11 =cv2.imread("../ImagenesRotadasO/97_R_N.jpg", 0) #Leemos la imagen a BN

img12 =cv2.imread("../ImagenesRotadasO/98_R_T.jpg", 0) #Leemos la imagen a BN
img13 =cv2.imread("../ImagenesRotadasO/98_R_D.jpg", 0) #Leemos la imagen a BN
img14 =cv2.imread("../ImagenesRotadasO/98_R_C.jpg", 0) #Leemos la imagen a BN
img15 =cv2.imread("../ImagenesRotadasO/98_R_N.jpg", 0) #Leemos la imagen a BN

img16 =cv2.imread("../ImagenesRotadasO/99_R_T.jpg", 0) #Leemos la imagen a BN
img17 =cv2.imread("../ImagenesRotadasO/99_R_D.jpg", 0) #Leemos la imagen a BN
img18 =cv2.imread("../ImagenesRotadasO/99_R_C.jpg", 0) #Leemos la imagen a BN
img19 =cv2.imread("../ImagenesRotadasO/99_R_N.jpg", 0) #Leemos la imagen a BN
'''

# Convertir las imagenes a un arreglo de numpy 

img00 = np.array(img0)
img11 = np.array(img1)
img22 = np.array(img2)
img33 = np.array(img3)
'''
img44 = np.array(img4)
img55 = np.array(img5)
img66 = np.array(img6)
img77 = np.array(img7)

img88 = np.array(img8)
img99 = np.array(img9)
img1010 = np.array(img10)
img1111 = np.array(img11)

img1212 = np.array(img12)
img1313 = np.array(img13)
img1414 = np.array(img14)
img1515 = np.array(img15)

img1616 = np.array(img16)
img1717 = np.array(img17)
img1818 = np.array(img18)
img1919 = np.array(img19)
'''

# APlanar el arreglo a un vector de 30,000 x 1 

img000 = img00.reshape([30000,1]) #Tamaño dinámico
img111 = img11.reshape([30000,1]) #Tamaño dinámico
img222 = img22.reshape([30000,1]) #Tamaño dinámico
img333 = img33.reshape([30000,1]) #Tamaño dinámico

'''
img444 = img44.reshape([30000,1]) #Tamaño dinámico
img555 = img55.reshape([30000,1]) #Tamaño dinámico
img666 = img66.reshape([30000,1]) #Tamaño dinámico
img777 = img77.reshape([30000,1]) #Tamaño dinámico

img888 = img88.reshape([30000,1]) #Tamaño dinámico
img999 = img99.reshape([30000,1]) #Tamaño dinámico
img101010 = img1010.reshape([30000,1]) #Tamaño dinámico
img111111 = img1111.reshape([30000,1]) #Tamaño dinámico

img121212 = img1212.reshape([30000,1]) #Tamaño dinámico
img131313 = img1313.reshape([30000,1]) #Tamaño dinámico
img141414 = img1414.reshape([30000,1]) #Tamaño dinámico
img151515 = img1515.reshape([30000,1]) #Tamaño dinámico

img161616 = img1616.reshape([30000,1]) #Tamaño dinámico
img171717 = img1717.reshape([30000,1]) #Tamaño dinámico
img181818 = img1818.reshape([30000,1]) #Tamaño dinámico
img191919 = img1919.reshape([30000,1]) #Tamaño dinámico
'''

# Dividir cada elemento entre 255 para mejor eficiencia

X0 = img000/255
X1 = img111/255
X2 = img222/255
X3 = img333/255

'''
X4 = img444/255
X5 = img555/255
X6 = img666/255
X7 = img777/255

X8 = img888/255
X9 = img999/255
X10 = img101010/255
X11 = img111111/255

X12 = img121212/255
X13 = img131313/255
X14 = img141414/255
X15 = img151515/255

X16 = img161616/255
X17 = img171717/255
X18 = img181818/255
X19 = img191919/255
'''

np.savetxt('a.txt',X0,fmt='%f')
np.savetxt('b.txt',X1,fmt='%f')
np.savetxt('c.txt',X2,fmt='%f')
np.savetxt('d.txt',X3,fmt='%f')

'''
np.savetxt('e.txt',X4,fmt='%f')
np.savetxt('f.txt',X5,fmt='%f')
np.savetxt('g.txt',X6,fmt='%f')
np.savetxt('h.txt',X7,fmt='%f')

np.savetxt('i.txt',X8,fmt='%f')
np.savetxt('j.txt',X9,fmt='%f')
np.savetxt('k.txt',X10,fmt='%f')
np.savetxt('l.txt',X11,fmt='%f')

np.savetxt('m.txt',X12,fmt='%f')
np.savetxt('n.txt',X13,fmt='%f')
np.savetxt('ñ.txt',X14,fmt='%f')
np.savetxt('o.txt',X15,fmt='%f')

np.savetxt('p.txt',X16,fmt='%f')
np.savetxt('q.txt',X17,fmt='%f')
np.savetxt('r.txt',X18,fmt='%f')
np.savetxt('s.txt',X19,fmt='%f')
'''