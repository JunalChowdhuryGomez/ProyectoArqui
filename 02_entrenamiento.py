import cv2
import os
import numpy as np

dataPath = 'C:/Users/ACER/Desktop/RF/datos'  # Ruta donde has almacenado las imágenes
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)
labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = os.path.join(dataPath, nameDir)
    print(f'Leyendo las imágenes de {nameDir}')

    for fileName in os.listdir(personPath):
        print(f'Rostros: {nameDir}/{fileName}')
        imagePath = os.path.join(personPath, fileName)
        image = cv2.imread(imagePath, 0)
        if image is None:
            continue
        image = cv2.resize(image, (150, 150))
        image = cv2.equalizeHist(image)
        facesData.append(image)
        labels.append(label)
        cv2.imshow('image', image)
        cv2.waitKey(10)
    label += 1

cv2.destroyAllWindows()

print("Número de etiquetas únicas:", len(set(labels)))
print("Número de imágenes:", len(facesData))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))
face_recognizer.write('modeloLBPHFace.xml')
print("Modelo almacenado...")
