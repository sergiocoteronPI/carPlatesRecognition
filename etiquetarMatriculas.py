
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
imgNombres = leerDatos("datasetOCR/")

txtOCR = leerDatos("labelOCR/", archivosPermitidos=["txt"])

nombreVetados = []
if len(txtOCR) == 1:
    with open(txtOCR[0], "r") as f:
        for line in f:
            nombreVetados.append(line.rstrip("\n").split(",")[0])
else:
    input("NADA QUE LEER")

for name, contador in zip(imgNombres, range(len(imgNombres))):

    if name in nombreVetados:
        continue

    img = cv2.imread(name)

    cv2.imshow("dfg", img)
    cv2.waitKey(1)

    matricula = input("Matricula " + str(contador) + "/" + str(len(imgNombres)) + ": ")

    if matricula=="":
        continue

    newMat = ""
    for letra in matricula:
        newMat += letra.upper()

    with open("labelOCR/label.txt", "a") as f:
        f.write(name + "," + str(newMat) + "\n")