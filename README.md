# carPlatesRecognition
OCR

classControlOCR.py -> Clase con todo lo necesario para el manejo del entrenamiento y del test. Esta clase es necesaria en cualquier otro script del reconocimiento de matrículas u OCR para matrículas.

train.py -> Programa para el entrenamiento.
test.py -> Programa para ver los resultados del entrenamiento.

mark1.py -> (hay que cambiarle el nombre) Tenemos la red neuronal funcion pérdida de decodificación...

readDataset.py -> Aquí están las funciones de lectura de datos y de crear los lotes para entrenar. Tratamiento de imágenes etc.

-------

Esas eran las funciones principales. Tenemos ahora los scripts que me han permitido crear la base de datos.

predataset.py -> Basicamente lee si existe ya un registro de matrículas en "label/" que tiene que teer "*.txt" de la forma
                  "car3_img/car3_0000000001.jpg,366,767,697,874"
                          ...................................
                  "nombreImagen, coordX, coordY, coordW, coordH"
                  después guarda los recortes de las matrículas en "predataset" (carpeta que debe ser creada)
