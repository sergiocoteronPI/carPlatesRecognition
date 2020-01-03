# ===================================================================================================================================================== #
#                                         Importamos las librerias que sean necesarias así como nuestras clases y funciones.
import tensorflow as tf

import os
import numpy as np
from random import shuffle

from classControlOCR import clasMatOcr
from readDataset import cargarLote, leerDatos, cargarTxt

from mark1 import mark1, lossFunction

import sys
# ===================================================================================================================================================== #


# ===================================================================================================================================================== #
#                                         Leemos el txt (o los txts) que nos dice donde está la imagen y cual es la etiqueta de esta. Luego cargaremos los datos con la funcion cargarLote
datasetLabelImgNames, datasetLabelImgLabel = cargarTxt("labelOCR/label.txt")
if datasetLabelImgNames == [] or datasetLabelImgLabel == []:
    input("Esto esta fatal hay que parar la ejecucion YA")
# ===================================================================================================================================================== #


# ===================================================================================================================================================== #
#                                         Cargamos el modelo o lo creamos en caso de no existir.
if os.path.exists(clasMatOcr.h5):
    
    #model = tf.keras.models.load_model(self_.h5, custom_objects={'loss_function': loss_function})

    x, h_out = mark1()
    model = tf.keras.Model(inputs=x, outputs=h_out)
    
    model.compile(loss=lossFunction,optimizer=tf.keras.optimizers.Adam(lr = 0.001))
    #model.compile(loss=loss_function,optimizer=tf.keras.optimizers.RMSprop(lr=0.0001,rho=0.9,epsilon=None,decay=0.0))

    print('')
    print(' ===== cargando modelo =====')
    print('')
    
    model.load_weights(clasMatOcr.h5)
    
else:
    
    x, h_out = mark1()
    model = tf.keras.Model(inputs=x, outputs=h_out)
    
    model.compile(loss=lossFunction,optimizer=tf.keras.optimizers.Adam(lr = 0.001))
    #model.compile(loss=loss_function,optimizer=tf.keras.optimizers.RMSprop(lr=0.0001,rho=0.9,epsilon=None,decay=0.0))

print('')
print(model.summary())
print('')
# ===================================================================================================================================================== #

imgArrayTrain, labelArrayTrain = cargarLote(clasMatOcr,\
                                            datasetLabelImgNames, datasetLabelImgLabel,\
                                            0,len(datasetLabelImgNames),\
                                            preProcess = True) # Así es como se cargan los lotes

try:
    model.fit(x = imgArrayTrain, y = labelArrayTrain,
            batch_size = clasMatOcr.batch_size,
            epochs=5,
            verbose=1)

    print('')
    print(' ===== salvando modelo =====')
    print('')

    tf.keras.models.save_model(model, clasMatOcr.h5)
except:

    print("")
    print("")
    guardar = input("Quieres guardar el modelo: ")
    print("")
    if guardar in ["s", "si", "y", "yes", "Y"]:

        print('')
        print(' ===== salvando modelo =====')
        print('')
        
        tf.keras.models.save_model(model, clasMatOcr.h5)
    else:
        sys.exit