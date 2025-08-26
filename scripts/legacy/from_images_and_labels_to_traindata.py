
import os
import random
import shutil

def dividir_datos(carpeta_imagenes, carpeta_etiquetas, carpeta_destino):
    # Crear carpetas de entrenamiento y validación
    os.makedirs(os.path.join(carpeta_destino, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(carpeta_destino, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(carpeta_destino, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(carpeta_destino, 'labels', 'val'), exist_ok=True)

    # Obtener la lista de archivos en la carpeta de imágenes
    imagenes = os.listdir(carpeta_imagenes)
    random.shuffle(imagenes)  # Mezclar aleatoriamente la lista de imágenes

    # Calcular el número de imágenes para el conjunto de entrenamiento y validación
    total_imagenes = len(imagenes)
    num_imagenes_entrenamiento = int(0.8 * total_imagenes)

    # Dividir las imágenes y moverlas a las carpetas correspondientes
    for i, imagen in enumerate(imagenes):
        # Obtener el nombre del archivo de la imagen y su correspondiente archivo de etiquetas
        nombre_imagen, extension_imagen = os.path.splitext(imagen)
        archivo_etiquetas = nombre_imagen + '.txt'

        # Comprobar si existe el archivo de etiquetas correspondiente
        ruta_imagen = os.path.join(carpeta_imagenes, imagen)
        ruta_etiquetas = os.path.join(carpeta_etiquetas, archivo_etiquetas)
        
        if os.path.exists(ruta_etiquetas):
            # Determinar la carpeta de destino (entrenamiento o validación)
            carpeta_destino_imagenes = 'train' if i < num_imagenes_entrenamiento else 'val'

            # Mover la imagen y su archivo de etiquetas a las carpetas correspondientes
            shutil.move(ruta_imagen, os.path.join(carpeta_destino, 'images', carpeta_destino_imagenes, imagen))
            shutil.move(ruta_etiquetas, os.path.join(carpeta_destino, 'labels', carpeta_destino_imagenes, archivo_etiquetas))
        else:
            print(f"No se encontró el archivo de etiquetas para la imagen: {imagen}. No se moverá.")



carpeta_imagenes =  'C:/Cosas/2025/IBERO2025/train_data_images'
carpeta_etiquetas = 'C:/Cosas/2025/IBERO2025/train_data_labels'
carpeta_destino =   'C:/Cosas/2025/IBERO2025/train_data'


dividir_datos(carpeta_imagenes, carpeta_etiquetas, carpeta_destino)

# Lleva desde las dos carpetas train_data_labels y train_data_images, y las guarda en una única carpeta listo para entrenar

# Ejecutar así: 
# python "C:\Cosas\2025\IBERO2025\scripts_python\from_images_and_labels_to_traindata.py"
