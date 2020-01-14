# ===================================================================================================================================================== #
#                                         Importamos las librerias que sean necesarias así como nuestras clases y funciones.
import tensorflow as tf

import cv2

import os
import numpy as np
from random import shuffle

from classControlOCR import clasMatOcr
from readDataset import cargarLote, leerDatos, cargarTxt

from mark1 import mark1, lossFunction
# ===================================================================================================================================================== #


# ===================================================================================================================================================== #
#                                         Creamos una clase para tener ahí los nombres del datset. Como se puede ver las imágenes deben estar en _path = clasMatOcr.rpi
datasetLabelImgNames, datasetLabelImgLabel = cargarTxt(clasMatOcr.rpi + "labelOCR/label.txt")
if datasetLabelImgNames == [] or datasetLabelImgLabel == []:
    input("Esto esta fatal hayq eu parar la ejecucion YA")
# ===================================================================================================================================================== #

from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

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


def traducir(a_traducir):
    frase = ''
    for i in a_traducir:
        if i == 200:
            break
        frase += clasMatOcr.dict[i]
    return frase
    
def decode(inputs, sequence_length):

    #return tf.nn.ctc_beam_search_decoder(inputs,sequence_length,beam_width=100,top_paths=1,merge_repeated=True)
    #return tf.keras.backend.ctc_decode(inputs,sequence_length,greedy=True,beam_width=100,top_paths=1)
    #return tf.nn.ctc_greedy_decoder(inputs,sequence_length,merge_repeated=True)

    return tf.nn.ctc_greedy_decoder(inputs, sequence_length=sequence_length)#features['seq_lens'])

imgArrayTrain, labelArrayTrain = cargarLote(clasMatOcr,\
                                            datasetLabelImgNames, datasetLabelImgLabel,\
                                            0,len(datasetLabelImgNames),\
                                            preProcess = False) # Así es como se cargan los lotes


### Código para desordenar las listas ###
### ================================= ###
for i in range(len(imgArrayTrain)):

    num1 = np.random.randint(len(imgArrayTrain))
    num2 = np.random.randint(len(imgArrayTrain))

    if num1 == num2:
        continue

    auxImgArrayTrain = imgArrayTrain[num1]
    auxLabelArrayTrain = labelArrayTrain[num1]

    imgArrayTrain[num1] = imgArrayTrain[num2]
    labelArrayTrain[num1] = labelArrayTrain[num2]

    imgArrayTrain[num2] = auxImgArrayTrain
    labelArrayTrain[num2] = auxLabelArrayTrain
### ================================= ###

precisionMedia = 0
for imagen, etiqueta, contador in zip(imgArrayTrain, labelArrayTrain, range(1,len(imgArrayTrain))):

    net_out_ = model.predict(x=np.array([imagen]))
    net_out_reorganized = np.transpose(net_out_, (1, 0, 2))
        
    decoded, logProb = decode(net_out_reorganized, 1*[32])        
    decoded = tf.to_int32(decoded[0])

    decoded_traducido = tf.keras.backend.eval(decoded)[0]

    frase_predicha = traducir(decoded_traducido)
    
    labelString = traducir(etiqueta)

    probs = tf.keras.backend.eval(logProb)[0][0]
    precisionMedia += similar(labelString, frase_predicha)

    if contador % 100 == 0:
        print("Etiqueta: ", labelString, " -- Prediccion: ", frase_predicha,\
            " -- Precision: ", "{:.2f}".format(similar(labelString, frase_predicha)),\
            " -- Accuracy: ", "{:.2f}".format(precisionMedia/contador))