import cv2
import os

videosPath = 'C:/Users/ACER/Desktop/RF/videos/entrenamiento'  # Ruta donde has almacenado los videos de entrenamiento
outputPath = 'C:/Users/ACER/Desktop/RF/datos'  # Ruta donde deseas guardar los fotogramas

def extract_frames(video_path, output_dir, person_name, interval=2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    frame_count = 0
    saved_count = 0
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                output_file = os.path.join(output_dir, f"{person_name}_{saved_count}.jpg")
                cv2.imwrite(output_file, face)
                saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from {video_path}")

# Recorrer todos los videos en la carpeta
for file_name in os.listdir(videosPath):
    if file_name.endswith('.mp4') or file_name.endswith('.avi'):
        person_name = os.path.splitext(file_name)[0]
        video_path = os.path.join(videosPath, file_name)
        person_output_dir = os.path.join(outputPath, person_name)
        extract_frames(video_path, person_output_dir, person_name)
