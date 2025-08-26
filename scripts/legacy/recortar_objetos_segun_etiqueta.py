import os
import cv2

def recortar_objetos(carpeta_origen, carpeta_destino):
    # Crear carpeta destino si no existe
    os.makedirs(carpeta_destino, exist_ok=True)

    # Listar archivos .txt de etiquetas en la carpeta origen
    archivos = [f for f in os.listdir(carpeta_origen) if f.endswith('.txt')]
    
    for archivo_txt in archivos:
        # Obtener el nombre base (sin extensión)
        nombre_base = os.path.splitext(archivo_txt)[0]
        ruta_imagen = os.path.join(carpeta_origen, nombre_base + '.jpg')
        ruta_txt = os.path.join(carpeta_origen, archivo_txt)

        # Verificar si existe la imagen correspondiente
        if not os.path.exists(ruta_imagen):
            print(f"No se encontró la imagen correspondiente para {archivo_txt}")
            continue

        # Leer la imagen
        img = cv2.imread(ruta_imagen)
        h, w, _ = img.shape

        # Leer líneas del archivo de etiquetas
        with open(ruta_txt, 'r') as f:
            lineas = f.readlines()

        for idx, linea in enumerate(lineas):
            partes = linea.strip().split()
            if len(partes) != 5:
                continue  # formato incorrecto

            # YOLO: class x_center y_center width height (valores normalizados)
            _, x_center, y_center, box_width, box_height = partes
            x_center = float(x_center)
            y_center = float(y_center)
            box_width = float(box_width)
            box_height = float(box_height)

            # Convertir a coordenadas absolutas
            x1 = int((x_center - box_width/2) * w)
            y1 = int((y_center - box_height/2) * h)
            x2 = int((x_center + box_width/2) * w)
            y2 = int((y_center + box_height/2) * h)

            # Ajustar para no salir de la imagen
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w-1, x2)
            y2 = min(h-1, y2)

            # Recortar y guardar
            recorte = img[y1:y2, x1:x2]
            nombre_recorte = f"{nombre_base}_{idx:03}.jpg"
            ruta_recorte = os.path.join(carpeta_destino, nombre_recorte)
            cv2.imwrite(ruta_recorte, recorte)

            print(f"Guardado recorte: {ruta_recorte}")

# Ejemplo de uso
if __name__ == "__main__":
    carpeta_origen = "C:/Cosas/2025/IBERO2025/dataset-hojas-jun2025"  # carpeta con jpg y txt
    carpeta_destino = "C:/Cosas/2025/IBERO2025/dataset-hojas-recortes" # carpeta donde guardar los recortes
    recortar_objetos(carpeta_origen, carpeta_destino)

# Descripción:
# Toma una carpeta de entrada con imágenes .jpg y etiquetas .txt
# Recorta las regiones según las etiquetas YOLO y guarda cada recorte como un nuevo archivo en carpeta_destino
# Ejemplo: 0000_000.jpg, 0000_001.jpg, ...

# Ejecutar así:
# python "C:/Cosas/2025/IBERO2025/scripts_python/recortar_objetos_segun_etiqueta.py"
