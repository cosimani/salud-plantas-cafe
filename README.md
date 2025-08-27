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

# Ver configuración actual (opcional)
yolo settings

# Pararse en la raiz del repo

# Apuntar el datasets_dir al repo clonado (¡ajusta la ruta a tu clon!)
yolo settings datasets_dir="."


```bash
# Desde la raíz del repo
yolo train model=yolov8s.pt data=configs/labels.yaml epochs=100 imgsz=640 batch=32
```

- Pesos de salida típicos: `runs/detect/train/weights/best.pt` (Ultralytics los crea automáticamente).

---

## 4) Predicción y recortes automáticos

Con un modelo entrenado (ruta a tus `best.pt`), generá recortes de hojas a partir de una carpeta de imágenes **no etiquetadas**:

```bash
python scripts/predict_and_crop_with_yolov8.py --weights runs/detect/train/weights/best.pt --source  data/yolo/images/test --out_dir data/crops --conf 0.50
```

Salidas:
- Recortes: `data/crops/*.jpg`
- Metadatos: `data/crops/crops_manifest.csv` (archivo con caja, confianza y origen)
- Imágenes anotadas: `data/predict/*.jpg`

---

## 5) Próximos pasos (WIP)

- **SAM** para segmentar cada recorte a PNG con alpha (fondo transparente).
- **Clasificación** de hojas: saludable vs afectada.

> Abrí un issue si querés priorizar estos pasos, o enviá un PR.

instalar pip install git+https://github.com/facebookresearch/segment-anything.git


---

## 6) Segmentación con SAM (carpeta única) → PNG RGBA

para ejecutar en GPU 

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

y verificar con

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"


debería mostrar algo así

True NVIDIA GeForce RTX 4060 Laptop GPU



```bash
# Ejemplo: segmentar los recortes de YOLO en data/crops → data/sam
python scripts/segment/sam_segment_single_smallcrops.py   --input data/crops   --output data/sam   --checkpoint checkpoints/sam_vit_b_01ec64.pth   --model vit_b    --max_width 256 --min_area_frac 0.20
```


- Entrada: `data/crops/*.jpg` (recortes por hoja).
- Salida: `data/sam/*.png` (fondo transparente).

---

## 7) Clasificación (saludable vs afectada)

### Opción A — Clasificador simple (incluido en este repo)

Usamos un **ResNet18** fine-tune con `torchvision` sobre un dataset tipo **ImageFolder**:


data/classify/CoffeeLeaves/
train/healthy | train/affected
val/healthy | val/affected



> Dataset de ejemplo (mini) en el repo.  
> Dataset **completo** disponible en **Releases** (ver abajo).

**Entrenar (GPU/CPU):**
```bash
# Entrena y guarda pesos en runs/classify/resnet18/best.pt
python scripts/classify/train_resnet.py \
  --data_dir data/classify/CoffeeLeaves \
  --model resnet18 \
  --epochs 20 \
  --batch_size 32 \
  --lr 0.001 \
  --img_size 384 \
  --use_pretrained \
  --no_feature_extraction  # fine-tune completo

```

Inferencia sobre las PNG segmentadas por SAM:

# Clasifica cada PNG de data/sam en healthy/affected
python scripts/classify/predict_resnet.py \
  --weights runs/classify/resnet18/best.pt \
  --input  data/sam \
  --output data/classify_output \
  --move


Parámetros útiles:

--feature_extraction (en vez de --no_feature_extraction) congela el backbone y sólo entrena el head.

--img_size 224/256/384 según GPU.

--seed 42 para reproducibilidad.



7) Clasificación (saludable vs afectada)

La clasificación se entrena con ResNet18 (torchvision) usando un dataset ImageFolder:


data/
└─ classify/
   ├─ train/
   │   ├─ healthy/
   │   └─ affected/
   └─ val/
       ├─ healthy/
       └─ affected/


Hay un mini set de ejemplo en el repo.
El dataset completo está en Releases (dataset-leaves-yolo-classify-v1.zip).
Descomprimí en la raíz del repo y quedará data/classify/... como arriba.

Entrenamiento (GPU recomendado, funciona en CPU):

# CMD/PowerShell en la raíz del repo (una sola línea)
python scripts/classify/train_resnet.py --data_dir data/classify --model resnet18 --epochs 20 --batch_size 32 --lr 0.001 --img_size 384 --use_pretrained --no_feature_extraction




Salida: runs/classify/resnet18/best.pt (mejor val acc).

Si te quedás sin VRAM, probá --img_size 224 y/o --batch_size 16.

Inferencia sobre las PNG segmentadas por SAM (por carpeta):



# Clasifica cada PNG de data/sam en healthy o affected y copia/mueve a subcarpetas
python scripts/classify/predict_resnet.py --weights runs/classify/resnet18/best.pt --input data/sam --output data/classify_output --move



Parámetros útiles:

--feature_extraction (en lugar de --no_feature_extraction) congela el backbone y solo entrena la cabeza.

--seed 42 para reproducibilidad.

--topk 3 en predicción para guardar CSV con top-k scores.



Resumen de uso

Entrenar (desde raíz del repo):


python scripts/classify/train_resnet.py --data_dir data/classify --model resnet18 --epochs 20 --batch_size 32 --lr 0.001 --img_size 384 --use_pretrained --no_feature_extraction


Clasificar los PNG de data/sam:


python scripts/classify/predict_resnet.py --weights runs/classify/resnet18/best.pt --input data/sam --output data/classify_output --move





pipeline


Imagen única:

python scripts\pipeline\analyze_image.py --image data\yolo\images\test\test_0000.jpg --yolo_weights runs\detect\train\weights\best.pt --clf_weights runs\classify\resnet18\best.pt --out_dir runs\pipeline --yolo_imgsz 960 --yolo_conf 0.40 --save_crops --save_sam --sam_checkpoint checkpoints\sam_vit_b_01ec64.pth --sam_model vit_b --sam_max_width 256 --sam_min_area_frac 0.20 --separate_by_class



Carpeta entera

python scripts\pipeline\analyze_image.py --dir data\yolo\images\test --yolo_weights runs\detect\train\weights\best.pt --clf_weights runs\classify\resnet18\best.pt --out_dir runs\pipeline --yolo_imgsz 960 --yolo_conf 0.40 --save_crops --save_sam --sam_checkpoint checkpoints\sam_vit_b_01ec64.pth --sam_model vit_b --sam_max_width 256 --sam_min_area_frac 0.20 --separate_by_class








el pipeline de analyze_image.py está armado así (en ese orden):

Detector (YOLOv8)

Usa tu modelo YOLOv8 (--yolo_weights) sobre la imagen o carpeta.

Devuelve las cajas (bboxes) de cada hoja.

Recorte (YOLO crops)

Con cada bbox se genera un recorte (.jpg).

Si activaste --save_crops, se guardan en runs/pipeline/crops/.

Segmentación (SAM) (si activaste --save_sam)

Cada crop se pasa al SAM para segmentar la hoja.

Se genera un PNG RGBA (con transparencia de fondo).

Si usaste --separate_by_class, ya se ordena en sam/healthy o sam/affected según la clasificación.

Clasificación (healthy vs affected)

La entrada del clasificador es la salida de SAM (el PNG segmentado).

Si por algún motivo SAM falla en un crop → se clasifica directamente el recorte de YOLO.

Se guarda en el CSV y en la imagen anotada (predict/*.jpg).
👉 Entonces sí: el flujo principal es exactamente el que describís:
YOLO → crops → SAM (segmentación) → Clasificación sobre PNG segmentado.