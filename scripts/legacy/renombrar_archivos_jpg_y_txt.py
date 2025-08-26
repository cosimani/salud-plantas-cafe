import os

def renombrar_archivos_jpg_y_txt(carpeta, primer_numero):
    # Obtener la lista de todos los archivos en la carpeta
    archivos = os.listdir(carpeta)
    
    # Filtrar archivos .jpg y .txt
    archivos_jpg = sorted([f for f in archivos if f.lower().endswith('.jpg')])
    archivos_txt = set([os.path.splitext(f)[0] for f in archivos if f.lower().endswith('.txt')])
    
    # Contadores para archivos con etiquetas y sin etiquetas
    contador_con_etiqueta = primer_numero
    contador_sin_etiqueta = 0

    for archivo_jpg in archivos_jpg:
        nombre_base = os.path.splitext(archivo_jpg)[0]
        ruta_jpg_original = os.path.join(carpeta, archivo_jpg)
        
        if nombre_base in archivos_txt:
            # Caso donde hay un archivo .txt correspondiente
            nuevo_nombre_base = f"{contador_con_etiqueta:04d}"
            ruta_jpg_nuevo = os.path.join(carpeta, f"{nuevo_nombre_base}.jpg")
            ruta_txt_original = os.path.join(carpeta, f"{nombre_base}.txt")
            ruta_txt_nuevo = os.path.join(carpeta, f"{nuevo_nombre_base}.txt")
            
            # Renombrar el archivo jpg y el txt
            os.rename(ruta_jpg_original, ruta_jpg_nuevo)
            os.rename(ruta_txt_original, ruta_txt_nuevo)
            
            print(f"Renombrado: {archivo_jpg} y {nombre_base}.txt -> {nuevo_nombre_base}.jpg y {nuevo_nombre_base}.txt")
            
            contador_con_etiqueta += 1
        else:
            # Caso donde no hay un archivo .txt correspondiente
            nuevo_nombre_jpg = f"sin_etiquetar_{contador_sin_etiqueta:04d}.jpg"
            ruta_jpg_nuevo = os.path.join(carpeta, nuevo_nombre_jpg)
            
            # Renombrar solo el archivo jpg
            os.rename(ruta_jpg_original, ruta_jpg_nuevo)
            
            print(f"Renombrado sin etiqueta: {archivo_jpg} -> {nuevo_nombre_jpg}")
            
            contador_sin_etiqueta += 1

# Ejemplo de uso
# carpeta = 'C:/Cosas/2025/AlgoLabs2025/entrenamientos/yolo-tag-digits-detector/train_data_images'
carpeta = 'C:/Cosas/2025/IBERO2025/Imagenes_hojas'

primer_numero = 0  # Número inicial para los archivos con etiquetas

renombrar_archivos_jpg_y_txt(carpeta, primer_numero)

# Primero colocamos todos los jpg (no usar png) con sus correspondientes txt y los renombra a ambos comenzando por primer_numero, ajustado a 4 dígitos

# Ejecutar así:
# python "C:\Cosas\2025\IBERO2025\scripts_python\renombrar_archivos_jpg_y_txt.py"

