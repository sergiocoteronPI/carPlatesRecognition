
import string
import cv2
import os

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

magicNumb = 30
imgNombres = leerDatos("baseDeDatos/testImgs/")

txtOCR = leerDatos("baseDeDatos/labelOCR/", archivosPermitidos=["txt"])

nombreVetados = []
if len(txtOCR) == 1:
    with open(txtOCR[0], "r") as f:
        for line in f:
            nombreVetados.append(os.path.basename(line.rstrip("\n").split(",")[0]))
else:
    input("NADA QUE LEER")


preMatricula = "primeraMatricula"
for name, contador in zip(imgNombres, range(len(imgNombres))):

    if os.path.basename(name) in nombreVetados:
        continue
    
    try:
        img = cv2.resize(cv2.imread(name), (200,90))

        cv2.imshow("dfg", img)
        cv2.waitKey(1)
    except:
        continue

    matricula = input("Matricula " + str(contador) + "/" + str(len(imgNombres)) + ": ")

    if matricula!="":
        preMatricula = matricula
    else:
        matricula = preMatricula

    if matricula=="*delete":
        os.remove(name)
        continue

    newMat = ""
    for letra in matricula:
        newMat += letra.upper()

    with open("baseDeDatos/labelOCR/label.txt", "a") as f:
        f.write("testImgs/" + os.path.basename(name) + "," + str(newMat) + "\n")