
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
txtNombres = leerDatos("label/", archivosPermitidos=["txt"])

preCont = 1
for name in txtNombres:

    vector = []
    with open(name, "r") as f:
        for line in f:
            linea = line.rstrip("\n").split(',')

            vector.append(linea)

    if vector == []:
        continue

    nombre = "image/" + vector[0][0]

    img = cv2.imread(nombre)

    cont = 1
    for line in vector:

        try:
            x,y,w,h = int(float(line[1])), int(float(line[2])), int(float(line[3])), int(float(line[4]))
            newImg = img[y-magicNumb:h+magicNumb, x-magicNumb:w+magicNumb,:]

            cv2.imwrite("predataset/imagen_" + str(preCont) + "_mat_" + str(cont) + ".jpg", newImg)
            cont+=1
        except:
            print("Error en imagen: " + name)

    preCont+=1