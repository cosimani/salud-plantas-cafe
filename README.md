# 🌱 Salud de Plantas de Café  
**Detección → Recorte → Segmentación SAM → Clasificación (Healthy vs Affected)**  

[![Python](https://img.shields.io/badge/python-3.10%7C3.11-blue.svg?logo=python)](https://www.python.org/)  
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)](https://github.com/ultralytics/ultralytics)  
[![SAM](https://img.shields.io/badge/Segment%20Anything-Meta-orange)](https://github.com/facebookresearch/segment-anything)  
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

Este repositorio contiene una **receta reproducible** para analizar hojas de café:  

1. 🟩 **YOLOv8** detecta hojas.  
2. ✂️ Se generan recortes de cada hoja.  
3. 🎨 **SAM (Segment Anything)** segmenta cada hoja con fondo transparente.  
4. 🔎 Clasificador final identifica hojas **Healthy** vs **Affected**.  

Funciona en **Windows / Linux / macOS** con Python 3.10/3.11.  
Soporta **GPU NVIDIA** (CUDA) o ejecución en **CPU**.  

---

## ⚙️ Instalación

Cloná el repositorio e instalá dependencias:  

```bash
git clone https://github.com/cosimani/salud-plantas-cafe.git
cd salud-plantas-cafe

# Crear entorno virtual
python -m venv .venv
# Activar
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

Instalar dependencias:  

- **CPU**  
  ```bash
  pip install -r requirements-cpu.txt
  ```
- **GPU (CUDA 11.8 / RTX)**  
  ```bash
  pip install -r requirements-gpu.txt
  ```

Verificá GPU:  
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📂 Estructura esperada

```plaintext
salud-plantas-cafe/
├─ configs/labels.yaml        # configuración YOLO (clases)
├─ checkpoints/               # modelos entrenado de SAM (se genera al ejecutar)
├─ data/                      # imágenes para entrenamiento yolo y clasificador (se debe descargar de Release)
├─ run/                       # modelos y predicciones de yolo y clasificación (se genera con el uso)
├─ scripts/                   # scripts principales
│  ├─ pipeline/analyze_image.py
│  ├─ predict_and_crop_with_yolov8.py
│  ├─ segment/sam_segment_single_smallcrops.py
│  └─ classify/{train_resnet.py, predict_resnet.py}
└─ README.md
```

- **runs/** → resultados (se ignora en Git)  
- **data/** → datasets locales (se ignora en Git)  

> Los datasets completos se publican como **Releases** de GitHub.  

---

## 🏋️ Entrenamiento YOLOv8

```bash

python scripts/train_yolov8.py

```

📦 Dataset YOLO disponible en **Releases**:  
👉 [dataset-leaves-yolo-classify-v1.zip](https://github.com/cosimani/salud-plantas-cafe/releases)

---

## 🔍 Predicción + Recortes

Generar recortes (`data/crops/`) y anotaciones (`runs/predict/`):

```bash
python scripts/predict_and_crop_with_yolov8.py   --weights checkpoints/yolov8_best.pt   --source data/yolo/images/test   --out_dir runs/crops   --conf 0.40
```

---

## 🎨 Segmentación con SAM

Segmentar recortes YOLO a PNG con fondo transparente:  

```bash
python scripts/segment/sam_segment_single_smallcrops.py   --input runs/crops   --output runs/sam   --checkpoint checkpoints/sam_vit_b_01ec64.pth   --model vit_b   --max_width 256   --min_area_frac 0.20
```

---

## 🧪 Clasificación (Healthy vs Affected)

Entrenar clasificador **ResNet18**:

```bash
python scripts/classify/train_resnet.py   --data data/classify   --epochs 20   --batch_size 32
```

Inferencia:

```bash
python scripts/classify/predict_resnet.py   --weights checkpoints/resnet18_best.pt   --input runs/sam   --output runs/classified
```

📦 Dataset clasificación disponible en **Releases**:  
👉 [dataset-leaves-yolo-classify-v1.zip](https://github.com/cosimani/salud-plantas-cafe/releases)

---

## 🔗 Pipeline completo

Un solo comando para todo el flujo:  

```bash
python scripts/pipeline/analyze_image.py   --dir data/yolo/images/test   --yolo_weights checkpoints/yolov8_best.pt   --clf_weights checkpoints/resnet18_best.pt   --out_dir runs/pipeline   --yolo_imgsz 960 --yolo_conf 0.40   --save_crops --save_sam   --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth   --sam_model vit_b --sam_max_width 256 --sam_min_area_frac 0.20   --separate_by_class
```

📊 Resultado:  
- Total de hojas detectadas  
- Healthy vs Affected  
- CSV + imágenes anotadas en `runs/pipeline/`  

---

## 📌 Créditos

- **YOLOv8** → [Ultralytics](https://github.com/ultralytics/ultralytics)  
- **SAM (Segment Anything)** → [Meta AI](https://github.com/facebookresearch/segment-anything)  
- **Lacunarity Pooling (opcional)** → [AVL Lab CVPRW 2024](https://github.com/Advanced-Vision-and-Learning-Lab/2024_V4A_Lacunarity_Pooling_Layer)  

---

## 📜 Licencia

Este proyecto se distribuye bajo licencia [MIT](LICENSE).  
¡Sentite libre de usarlo, adaptarlo y contribuir! 🤝
