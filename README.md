# Salud de Plantas de Café — Receta rápida (dataset → YOLOv8 → recortes)

Esta receta te lleva de cero a:
1) preparar dataset para YOLOv8,  
2) entrenar,  
3) predecir y **recortar automáticamente** cada hoja detectada.

> Probado en Windows (Acer Nitro) con GPU NVIDIA.

---

## 0) Requisitos (Windows / GPU)

- Python 3.11.6: <https://www.python.org/downloads/release/python-3116/>
- (Opcional) CUDA Toolkit 11.8: <https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local>

Crear/activar entorno virtual (se reutiliza siempre):

```powershell
cd C:\Cosas\EntornosVirtuales
python -m venv env
C:\Cosas\EntornosVirtuales\env\Scripts\activate
pip install ultralytics
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verificación GPU:
```powershell
python - << "PY"
import torch
print("CUDA OK?" , torch.cuda.is_available())
PY
```

> Para reactivar el entorno en cualquier momento:
> `C:\Cosas\EntornosVirtuales\env\Scripts\activate`

---

## 1) Preparar dataset (renombrar, ordenar, split)

### 1.1 Renombrar imágenes y etiquetas (IBERO → nombres estándar)
- Carpeta fuente con JPG + TXT (makesense):  
  `C:\Cosas\2025\IBERO2025\Imagenes_hojas`

- Ejecutar:
```powershell
python "C:\Cosas\2025\IBERO2025\scripts_python\renombrar_archivos_jpg_y_txt.py"
```
Esto genera nombres consecutivos y marca los no etiquetados como `sin_etiquetar_XXXX.jpg`.

### 1.2 Llevar a `train_data_images` / `train_data_labels`
```powershell
python "C:\Cosas\2025\IBERO2025\scripts_python\from_traindata_to_images_and_labels.py"
```

### 1.3 Armar estructura YOLO (train/val)
```powershell
python "C:\Cosas\2025\IBERO2025\scripts_python\from_images_and_labels_to_traindata.py"
```

La estructura final queda así:
```
C:\Cosas\2025\IBERO2025\train_data\
  images\train, images\val
  labels\train, labels\val
```

---

## 2) Entrenar YOLOv8 (detección de hojas)

Colocar el archivo de datos (puede estar aquí o en `configs/labels.yaml`):

```yaml
# C:\Cosas\2025\IBERO2025\train_data\labels.yaml
path: C:\Cosas\2025\IBERO2025\train_data
train: C:\Cosas\2025\IBERO2025\train_data\images\train
val:   C:\Cosas\2025\IBERO2025\train_data\images\val
nc: 1
names: [hoja]
```

Entrenar:
```powershell
cd C:\Cosas\2025\IBERO2025\train_data
yolo train model=yolov8s.pt data=labels.yaml epochs=100 imgsz=320 batch=32
```

> Pesos resultantes (revisar el nombre exacto que crea Ultralytics):
> `C:\Cosas\2025\IBERO2025\train_data\runs\detect\train\weights\best.pt`

---

## 3) Predicción y **recortes automáticos** con YOLOv8

- Carpeta con imágenes **no etiquetadas**:
  `C:\Cosas\2025\IBERO2025\hojas_sin_etiquetar`

- Script (incluido en este repo):  
  `scripts\predict_and_crop_with_yolov8.py`

Ejecutar:
```powershell
python C:\Cosas\2025\IBERO2025\salud-plantas-cafe\scripts\predict_and_crop_with_yolov8.py ^
  --weights C:\Cosas\2025\IBERO2025\train_data\runs\detect\train\weights\best.pt ^
  --source  C:\Cosas\2025\IBERO2025\hojas_sin_etiquetar ^
  --out_dir C:\Cosas\2025\IBERO2025\train_data\crops ^
  --conf 0.25
```

Salidas:
- recortes: `C:\Cosas\2025\IBERO2025\train_data\crops\*.jpg`
- metadatos: `C:\Cosas\2025\IBERO2025\train_data\crops\crops_manifest.csv`

> Tip: si querés descartar detecciones muy pequeñas, podemos añadir un filtro por **área mínima** de bounding box o un **padding** para que el recorte incluya borde.

---

## 4) Próximos pasos (SAM + clasificación)

- **SAM:** correr sobre `train_data\crops` para obtener PNG con **fondo transparente**.  
- **Clasificación:** entrenar un clasificador (saludable vs afectada) sobre PNG segmentados.

> Cuando termines esta parte, seguimos con scripts para SAM + clasificación final.
