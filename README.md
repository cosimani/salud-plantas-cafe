# Salud de Plantas de Café
Detección → Recorte → (próx.) Segmentación SAM → (próx.) Clasificación

Este repositorio contiene una **receta reproducible** para:
1. Entrenar **YOLOv8** para detectar hojas de plantas de café.
2. Predecir con YOLOv8 y **recortar automáticamente** cada hoja detectada.
3. (Próximo) Ejecutar **SAM** para segmentar las hojas a PNG con fondo transparente.
4. (Próximo) Clasificar hojas **saludables** vs **afectadas**.

> Funciona en Windows, Linux y macOS. Requiere Python 3.10/3.11. Para acelerar con GPU NVIDIA, instalá PyTorch con CUDA según tu plataforma.

---

## 1) Instalación

```bash
# Clonar el repo
git clone https://github.com/cosimani/salud-plantas-cafe.git
cd salud-plantas-cafe

# Crear entorno virtual (Python 3.10/3.11)
python -m venv .venv
# Windows
. .venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# Instalar dependencias del proyecto
pip install -r requirements.txt

# Instalar PyTorch según tu sistema/GPU (elige tu comando en https://pytorch.org/get-started/locally/)
# Ejemplo CPU-only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> Verificá la GPU (opcional):
> ```python
> python -c "import torch; print('CUDA disponible?', torch.cuda.is_available())"
> ```

---

## 2) Estructura esperada de datos

Usaremos un dataset en formato **YOLO** dentro de `data/yolo` **(rutas relativas)**:

```
data/
└─ yolo/
   ├─ images/
   │  ├─ train/   # tus imágenes de entrenamiento
   │  └─ val/     # tus imágenes de validación
   └─ labels/
      ├─ train/   # txt YOLO por imagen (clase x_c y_c w h normalizados)
      └─ val/
```

- Clases definidas en `configs/labels.yaml` (aquí solo 1 clase: `leaf`).

Si ya tenés imágenes **sin etiquetar**, podés dejarlas en cualquier carpeta (p. ej. `data/unlabeled/`) para luego generar recortes con el modelo entrenado.

> Nota: En `scripts/legacy/` se incluyen utilidades que usó el equipo IBERO para renombrar y organizar datasets; son **opcionales** y pueden requerir ajustes menores de rutas.

---

## 3) Entrenamiento YOLOv8

```bash
# Desde la raíz del repo
yolo train model=yolov8s.pt data=configs/labels.yaml epochs=100 imgsz=320 batch=32
```

- Pesos de salida típicos: `runs/detect/train/weights/best.pt` (Ultralytics los crea automáticamente).

---

## 4) Predicción y recortes automáticos

Con un modelo entrenado (ruta a tus `best.pt`), generá recortes de hojas a partir de una carpeta de imágenes **no etiquetadas**:

```bash
python scripts/predict_and_crop_with_yolov8.py \
  --weights runs/detect/train/weights/best.pt \
  --source  data/unlabeled \
  --out_dir data/crops \
  --conf 0.25
```

Salidas:
- Recortes: `data/crops/*.jpg`
- Metadatos: `data/crops/crops_manifest.csv` (archivo con caja, confianza y origen)

**Flags útiles** (próximas versiones):
- `--min_area`: descartar cajas muy pequeñas
- `--pad`: agregar margen alrededor del recorte

---

## 5) Próximos pasos (WIP)

- **SAM** para segmentar cada recorte a PNG con alpha (fondo transparente).
- **Clasificación** de hojas: saludable vs afectada.

> Abrí un issue si querés priorizar estos pasos, o enviá un PR.

---

## 6) Segmentación con SAM (carpeta única) → PNG RGBA

```bash
# Ejemplo: segmentar los recortes de YOLO en data/crops → data/sam
python scripts/segment/sam_segment_single.py   --input data/crops   --output data/sam   --checkpoint checkpoints/sam_vit_b_01ec64.pth   --model vit_b   --max_width 1280   --min_area_frac 0.002
```

- Entrada: `data/crops/*.jpg` (recortes por hoja).
- Salida: `data/sam/*.png` (fondo transparente).

---

## 7) Clasificación (saludable vs afectada)

### Opción A: clasificador simple (incluido más adelante)
*(Próxima sección; baseline con sklearn o torch).*

### Opción B: **framework Lacunarity** (repo externo)  
Si querés replicar exactamente el pipeline que usaste (Pooling por Lacunaridad: CVPRW 2024), usá el repo externo y copiá adentro los archivos adaptados que están en `scripts/classify/external_lacunarity/` de este repo.

Pasos:
1. Clonar el repo oficial (o añadirlo como submódulo):
   ```bash
   git clone https://github.com/Advanced-Vision-and-Learning-Lab/2024_V4A_Lacunarity_Pooling_Layer.git
   cd 2024_V4A_Lacunarity_Pooling_Layer
   pip install -r requirements.txt
   ```
2. Copiar los archivos adaptados desde **este** repo a la misma estructura del repo externo:
   - `Datasets/Get_transform.py`
   - `Prepare_Data.py`
   - `Demo_Parameters.py` (añade `CoffeeLeaves` = 2 clases)
   - `demo.py` (entrenamiento FT)
   - `View_Results.py` (resúmenes y curvas)
   - `classify_folder.py` (inferencia y volcado por carpetas)
3. Estructurar tu dataset `CoffeeLeaves` (ImageFolder):
   ```
   2024_V4A_Lacunarity_Pooling_Layer/
     Datasets/CoffeeLeaves/
       train/healthy|affected
       val/healthy|affected
       test/healthy|affected  # opcional (si no, usa val)
   ```
4. Entrenar (ejemplo ResNet18 + MS_Lacunarity):
   ```bash
   python demo.py --data_selection 4 --pooling_layer 6 --agg_func 1 --model resnet18      --use_pretrained --no-feature_extraction --num_epochs 20 --earlystoppping 10      --lr 0.001 --train_batch_size 32 --val_batch_size 32 --test_batch_size 32 --resize_size 384 --use-cuda
   ```
5. Clasificar PNG segmentados con los **Best_Weights.pt** más recientes:
   ```bash
   python classify_folder.py --input "<ruta a data/sam>" --output "classified_output" --move
   # o indicando pesos explícitos:
   # --weights "<.../Saved_Models/MS_Lacunarity/global/Fine_Tuning/CoffeeLeaves/resnet18/Run_1/Best_Weights.pt>"
   ```

---

## 📦 Dataset YOLO (estructura + ejemplos)

Este repositorio incluye ya creada la estructura `data/yolo` con un **mini set de ejemplo**:

```
data/yolo/
  images/
    train/   (8 imágenes de ejemplo con sus labels)
    val/     (2 imágenes de ejemplo con sus labels)
    test/    (5 imágenes de ejemplo sin labels)
  labels/
    train/   (8 .txt YOLO)
    val/     (2 .txt YOLO)
configs/
  labels.yaml
```

- El archivo `configs/labels.yaml` ya apunta a estas rutas (repo-relativas).
- Clase única: `leaf`.

### ▶️ Probar con el mini set de ejemplo
Para verificar que todo funciona (entrenamiento corto):
```bash
yolo train model=yolov8s.pt data=configs/labels.yaml epochs=1 imgsz=320 batch=8
```



### ⬇️ Descargar el dataset completo
El dataset completo **no está en el repo** para mantenerlo liviano.  
Podés bajarlo desde la sección **[Releases](../../releases)** como `dataset-hojas-cafe-yolov8-v1.zip`.

1. Descargar `dataset-hojas-cafe-yolov8-v1.zip`.
2. Descomprimir en la raíz del repo, sobrescribiendo `data/yolo`:

```powershell
# Windows (PowerShell)
Expand-Archive -Path .\dataset-hojas-cafe-yolov8-v1.zip -DestinationPath .
```

```bash
# Linux/macOS
unzip dataset-hojas-cafe-yolov8-v1.zip -d .
```

### ➕ Agregar más imágenes etiquetadas
1. Copiar tus `.jpg/.png` a:
   - `data/yolo/images/train` o `data/yolo/images/val`
2. Copiar los `.txt` YOLO (mismo nombre base) a:
   - `data/yolo/labels/train` o `data/yolo/labels/val`
3. Entrenar de nuevo:
```bash
yolo train model=yolov8s.pt data=configs/labels.yaml epochs=100 imgsz=320 batch=32
```

### 🔬 Usar el conjunto de test
- Si tus imágenes de `images/test` no tienen labels, podés evaluar visualmente:
```bash
yolo predict model=runs/detect/train/weights/best.pt source=data/yolo/images/test
```

- O generar recortes automáticos para alimentar a SAM:
```bash
python scripts/predict_and_crop_with_yolov8.py \
  --weights runs/detect/train/weights/best.pt \
  --source  data/yolo/images/test \
  --out_dir data/crops
```
