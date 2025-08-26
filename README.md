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
. .\.venv\Scripts\activate
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
