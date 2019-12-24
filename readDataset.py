
import numpy as np
import string
import cv2
import os

def retocar(clasMatOcr, img):

    zeros = np.zeros([clasMatOcr.dim_fil,clasMatOcr.dim_col])
    im_sha_1, im_sha_2 = img.shape
    if im_sha_1 >= clasMatOcr.dim_fil:
        if im_sha_2 >= clasMatOcr.dim_col:
            try:
                zeros = cv2.resize(img,(clasMatOcr.dim_col,clasMatOcr.dim_fil))
            except:
                return None
        else:
            try:
                zeros[:,0:im_sha_2] = cv2.resize(img,(im_sha_2,clasMatOcr.dim_fil))
            except:
                return None
    elif im_sha_2 >= clasMatOcr.dim_col:
        try:
            zeros[0:im_sha_1,:] = cv2.resize(img,(clasMatOcr.dim_col,im_sha_1))
        except:
            return None
    else:
        zeros[0:im_sha_1, 0:im_sha_2] = img
    
    return zeros

def leerDatos(_path, ctrlArchPerm = True ,archivosPermitidos = ["jpg","jpeg","png","JPG"]):

    filesNomb = []

    for ruta, _, ficheros in os.walk(_path):
        for nombreFichero in ficheros:
            rutComp = os.path.join(ruta, nombreFichero)

            if ctrlArchPerm:
                for arcPerm in archivosPermitidos:
                    if rutComp.endswith(arcPerm):
                        filesNomb.append(rutComp)
            else:
                filesNomb.append(rutComp)

    return filesNomb

def cargarLote(clasMatOcr, filesNomb, desde, hasta):

    labelArray = []
    imgArray = []

    for name in filesNomb[desde : hasta]:
        
        try:

            imgArray.append(retocar(clasMatOcr, cv2.imread(name,0)).astype("uint8"))

            finalName = []
            nameRev = os.path.basename(name).split('.')[0]
            for letra in nameRev:
                if letra.upper() in clasMatOcr.dict:
                    finalName .append(clasMatOcr.dict.index(letra.upper()))
            for _ in range(len(finalName), 10):
                finalName.append(200)
                    
            labelArray.append(finalName)
        except:
            return None, None

    return np.array(imgArray), np.array(labelArray)
    