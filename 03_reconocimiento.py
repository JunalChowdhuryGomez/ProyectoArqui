import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import Listbox, Scrollbar
from threading import Thread

# Ruta donde has almacenado Data y el modelo entrenado
dataPath = 'C:/Users/ACER/Desktop/RF/datos'
modelPath = 'modeloLBPHFace.xml'
videoPath = 'C:/Users/ACER/Desktop/RF/videos/reconocimiento/final.mp4'  # Ruta del video para reconocimiento
peopleList = os.listdir(dataPath)
print('Lista de personas:', peopleList)

# Cargar el modelo entrenado
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(modelPath)

# Variable global para rastrear las personas reconocidas
recognized_people = set()

def show_file_access(person_name):
    user_files_dir = os.path.join('user_files', person_name)
    if not os.path.exists(user_files_dir):
        os.makedirs(user_files_dir)
    
    files = os.listdir(user_files_dir)
    
    # Crear ventana para mostrar los archivos
    root = tk.Tk()
    root.title(f"Archivos de {person_name}")
    root.geometry("400x300")
    
    scrollbar = Scrollbar(root)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    listbox = Listbox(root, yscrollcommand=scrollbar.set)
    for file in files:
        listbox.insert(tk.END, file)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    scrollbar.config(command=listbox.yview)
    
    root.mainloop()
    recognized_people.remove(person_name)# Quitar la persona del set cuando la ventana se cierra
    

# Iniciar la captura de video
cap = cv2.VideoCapture(videoPath)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


fps = cap.get(cv2.CAP_PROP_FPS)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in detected_faces:
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        rostro = cv2.equalizeHist(rostro)
        result = face_recognizer.predict(rostro)

        label = result[0]
        confidence = result[1]

        print(f'Persona detectada: {peopleList[label]} con confianza {confidence}')

        if confidence < 65:
            name = peopleList[label]
            print(f"Reconocido: {name} con confianza {confidence}")
            cv2.putText(frame, name, (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Añadir un marco verde alrededor del rostro reconocido

            if name not in recognized_people:
                recognized_people.add(name)
                #Thread(target=show_file_access, args=(name,)).start()
        else:
            cv2.putText(frame, 'Persona Desconocida', (x, y-25), 2, 1.1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) 
    
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('w'):
        break

cap.release()
cv2.destroyAllWindows()
