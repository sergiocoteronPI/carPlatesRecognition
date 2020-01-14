# ===================================================================================================================================================== #
#                                         Importamos las librerias que sean necesarias así como nuestras clases y funciones.
import tensorflow as tf

import cv2

import os
import numpy as np
from random import shuffle

from classControlOCR import clasMatOcr
from readDataset import cargarLote, leerDatos, cargarTxt, retocar

from mark1 import mark1, lossFunction
# ===================================================================================================================================================== #

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

imagesTest = leerDatos(clasMatOcr.rpi + "testImgs")

precisionMedia = 0
for name, contador in zip(imagesTest, range(1,len(imagesTest))):

    imagen = retocar(clasMatOcr, cv2.imread(name,0), preProcess = False).astype("uint8")

    net_out_ = model.predict(x=np.array([imagen]))
    net_out_reorganized = np.transpose(net_out_, (1, 0, 2))
        
    decoded, logProb = decode(net_out_reorganized, 1*[32])        
    decoded = tf.to_int32(decoded[0])

    decoded_traducido = tf.keras.backend.eval(decoded)[0]

    frase_predicha = traducir(decoded_traducido)


    print("Prediccion: ", frase_predicha)

    cv2.imshow("img", cv2.resize(imagen, (256,98)))
    cv2.waitKey(0)