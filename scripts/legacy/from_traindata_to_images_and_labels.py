import os
import shutil

def organizar_archivos(carpeta_train_data,carpeta_train_data_images,carpeta_train_data_labels):
    
    # Crear las carpetas si no existen
    os.makedirs(carpeta_train_data_images, exist_ok=True)
    os.makedirs(carpeta_train_data_labels, exist_ok=True)
    
    # Explorar todas las subcarpetas
    for raiz, _, archivos in os.walk(carpeta_train_data):
        for archivo in archivos:
            ruta_completa = os.path.join(raiz, archivo)
            # Verificar la extensión del archivo
            if archivo.endswith('.jpg'):
                # Mover archivos .jpg a la carpeta de imágenes
                shutil.move(ruta_completa, carpeta_train_data_images)
            elif archivo.endswith('.txt'):
                # Mover archivos .txt a la carpeta de etiquetas
                shutil.move(ruta_completa, carpeta_train_data_labels)

carpeta_train_data =        'C:/Cosas/2025/IBERO2025/train_data'
carpeta_train_data_images = 'C:/Cosas/2025/IBERO2025/train_data_images'
carpeta_train_data_labels = 'C:/Cosas/2025/IBERO2025/train_data_labels'



organizar_archivos(carpeta_train_data,carpeta_train_data_images,carpeta_train_data_labels)

# Mueve desde la carpeta listo para entrenar, a las dos carpetas train_data_images y train_data_labels con todo junto

# Ejecutar así: 
# python "C:\Cosas\2025\IBERO2025\scripts_python\from_traindata_to_images_and_labels.py"

