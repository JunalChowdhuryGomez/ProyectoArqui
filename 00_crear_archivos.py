import os
import random

def create_random_files(base_folder, folders, extensions, num_files):
    for folder in folders:
        # Crear archivos respectivammente
        folder_path = os.path.join(base_folder, folder)
        os.makedirs(folder_path, exist_ok=True)
        
        for _ in range(num_files):
            # Elegir el tipo de archivo segun su extension
            ext = random.choice(extensions)
            # Crear archivos de acuerdo al nombre
            filename = f"{folder}_{random.randint(1, 100)}{ext}"
            file_path = os.path.join(folder_path, filename)
            # Crea un archivo vacio con el nombre especificado
            with open(file_path, 'w') as file:
                file.write(f"Este es un archivo de prueba llamado {filename} en la carpeta {folder}.")

# Base folder where the user folders are located
base_folder = 'user_files'

# List of user folders
folders = ['cristian', 'junal','junal_','cristian','junal__']

# List of possible file extensions
extensions = ['.pdf', '.docx', '.txt', '.xlsx']

# Number of files to create in each folder
num_files = 5

# Create the random files
create_random_files(base_folder, folders, extensions, num_files)