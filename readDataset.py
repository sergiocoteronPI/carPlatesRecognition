
import numpy as np
import string
import cv2
import os

from random import shuffle

def retocar(clasMatOcr, img, preProcess = False):

    zeros = np.zeros([clasMatOcr.dim_fil,clasMatOcr.dim_col])
    im_sha_1, im_sha_2 = img.shape

    if preProcess:

        #cv2.imshow("imgOrg", cv2.resize(img, (200,64)))

        img = img[np.random.randint(15):im_sha_1 - np.random.randint(15), np.random.randint(15):im_sha_2 - np.random.randint(15)]
        im_sha_1, im_sha_2 = img.shape

        img = rotate_bound(img, np.random.randint(-30,30))

        #cv2.imshow("imgFinal", cv2.resize(img, (200,64)))
        #cv2.waitKey(0)

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

def rotate_bound(image, angle):

    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    return cv2.warpAffine(image, M, (nW, nH))

def cargarTxt(_path):

    datasetLabelImgNames = []
    datasetLabelImgLabel = []

    todasLasLineas = []
    with open(_path, "r") as f:
        for line in f:
            todasLasLineas.append(line)

    shuffle(todasLasLineas)

    for line in todasLasLineas:
        datasetLabelImgNames.append(line.rstrip("\n").split(",")[0])
        datasetLabelImgLabel.append(line.rstrip("\n").split(",")[1])

    return datasetLabelImgNames, datasetLabelImgLabel

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

def cargarLote(clasMatOcr, datasetLabelImgNames, datasetLabelImgLabel, desde, hasta, preProcess = False):

    labelArray = []
    imgArray = []

    for _ in range(1):

        for name, nameRev in zip(datasetLabelImgNames[desde : hasta], datasetLabelImgLabel[desde : hasta]):
            
            try:
                imgArray.append(retocar(clasMatOcr, cv2.imread(clasMatOcr.rpi + name,0), preProcess = preProcess).astype("uint8"))
                
                finalName = []
                #nameRev = os.path.basename(name).split('.')[0]
                for letra in nameRev:
                    if letra.upper() in clasMatOcr.dict:
                        finalName .append(clasMatOcr.dict.index(letra.upper()))
                for _ in range(len(finalName), 10):
                    finalName.append(200)
                        
                labelArray.append(finalName)
            except:
                continue

    nad = clasMatOcr.batch_size*int(len(imgArray) / clasMatOcr.batch_size)

    return np.array(imgArray[:nad]), np.array(labelArray[:nad])
    