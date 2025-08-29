# scripts/detect_and_crop_yolov8.py
# -*- coding: utf-8 -*-
"""
Predice hojas con YOLOv8, genera recortes y guarda una imagen anotada por input.

Ejecución sin parámetros:
    python scripts/detect_and_crop_yolov8.py

Por defecto:
- Pesos:  models/best.pt
- Fuente: data/yolo/images/test/
- Salida: runs/predict/<YYYYmmdd-HHMMSS>/{annotated,crops}
- Copia adicional de {annotated,crops} en: runs/predict/latest/{annotated,crops}
- Device: GPU 0 si está disponible, caso contrario CPU
"""

import os
import csv
import uuid
import cv2
import shutil
from datetime import datetime

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    raise SystemExit(
        "Faltan dependencias. Instalá ultralytics y torch, p. ej.:\n"
        "  pip install -r requirements-gpu.txt   # o requirements-cpu.txt"
    )

# ---------- Defaults ----------
DEFAULT_WEIGHTS = os.path.join("models", "best.pt")
DEFAULT_SOURCE  = os.path.join("data", "yolo", "images", "test")
DEFAULT_CONF    = 0.40
DEFAULT_IOU     = 0.45
DEFAULT_IMGSZ   = 960

# ---------- Filtro de tamaño mínimo (px) ----------
# Acepta si:
#   - Rectangular: un lado >= MIN_LONG y el otro >= MIN_SHORT (en cualquier orden), o
#   - Cuadrado/Grande: ambos lados >= MIN_SQ.
MIN_SHORT = 40   # lado corto para formato rectangular
MIN_LONG  = 80   # lado largo para formato rectangular
MIN_SQ    = 50   # mínimo para formato ~cuadrado


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def crop_xyxy(img, xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def draw_box(img, xyxy, label_text, color=(0, 200, 255), thickness=2):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_text = max(0, y1 - 8)
    cv2.rectangle(img, (x1, y_text - th - 4), (x1 + tw + 6, y_text + baseline), color, -1)
    cv2.putText(img, label_text, (x1 + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def safe_copytree(src: str, dst: str) -> None:
    """Copia el contenido de src a dst (creando dst si no existe). No borra el padre."""
    ensure_dir(dst)
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target_root = os.path.join(dst, rel) if rel != "." else dst
        ensure_dir(target_root)
        for d in dirs:
            ensure_dir(os.path.join(target_root, d))
        for f in files:
            s = os.path.join(root, f)
            t = os.path.join(target_root, f)
            shutil.copy2(s, t)


def install_latest_atomic(run_root: str) -> None:
    """
    Instala runs/predict/latest como copia atómica de run_root:
    - Crea runs/predict/.latest_tmp
    - Copia {annotated,crops} adentro
    - Renombra .latest_tmp -> latest (swap), sin borrar runs/predict
    """
    predict_root = os.path.join("runs", "predict")
    ensure_dir("runs")
    ensure_dir(predict_root)

    tmp = os.path.join(predict_root, ".latest_tmp")
    latest = os.path.join(predict_root, "latest")

    # limpiar tmp si quedó de un intento anterior
    if os.path.isdir(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    elif os.path.islink(tmp) or os.path.isfile(tmp):
        try:
            os.unlink(tmp)
        except Exception:
            pass

    ensure_dir(tmp)
    # copiar solo carpetas relevantes
    for sub in ("annotated", "crops"):
        src_sub = os.path.join(run_root, sub)
        if os.path.isdir(src_sub):
            dst_sub = os.path.join(tmp, sub)
            safe_copytree(src_sub, dst_sub)

    # eliminar latest previo sin tocar el padre
    if os.path.islink(latest) or os.path.isfile(latest):
        try:
            os.unlink(latest)
        except Exception:
            pass
    elif os.path.isdir(latest):
        shutil.rmtree(latest, ignore_errors=True)

    # swap atómico
    os.replace(tmp, latest)  # en Windows también funciona para mover/renombrar


def is_valid_crop_size(w: int, h: int) -> bool:
    """
    Acepta si:
      - Rectangular: un lado >= MIN_LONG y el otro >= MIN_SHORT, en cualquier orden, o
      - Cuadrado/Grande: ambos lados >= MIN_SQ.
    """
    a, b = sorted((w, h))  # a = corto, b = largo
    return (a >= MIN_SHORT and b >= MIN_LONG) or (a >= MIN_SQ and b >= MIN_SQ)


def main():
    weights = DEFAULT_WEIGHTS
    source  = DEFAULT_SOURCE
    conf    = DEFAULT_CONF
    iou     = DEFAULT_IOU
    imgsz   = DEFAULT_IMGSZ

    # Device auto
    device = "0" if torch.cuda.is_available() else "cpu"

    # Asegurar raíces que NO deben borrarse
    ensure_dir("runs")
    ensure_dir(os.path.join("runs", "predict"))

    # Estructura de salida con timestamp
    run_id   = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = os.path.join("runs", "predict", run_id)
    crops_dir = os.path.join(base_dir, "crops")
    ann_dir   = os.path.join(base_dir, "annotated")
    ensure_dir(crops_dir)
    ensure_dir(ann_dir)

    # CSV de manifiesto de recortes (incluye descartados)
    csv_path = os.path.join(crops_dir, "crops_manifest.csv")
    fcsv = open(csv_path, "w", newline="", encoding="utf-8")
    wr = csv.writer(fcsv)
    wr.writerow([
        "orig_file", "crop_file", "status", "reason", "conf",
        "x1", "y1", "x2", "y2", "bbox_w", "bbox_h",
        "img_w", "img_h", "class_id", "class_name"
    ])

    # Cargar modelo
    model = YOLO(weights)
    names = getattr(model, "names", None) or {0: "leaf"}

    print("\n=== Predicción + Recortes (YOLOv8) ===")
    print(f"weights: {weights}")
    print(f"source:  {source}")
    print(f"conf:    {conf}")
    print(f"iou:     {iou}")
    print(f"imgsz:   {imgsz}")
    print(f"device:  {device}")
    print(f"out:     {base_dir}")
    print("=======================================\n")

    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        stream=True,
        verbose=True
    )

    total_imgs = 0
    total_saved = 0
    total_discarded = 0

    for r in results:
        img = r.orig_img
        if img is None:
            continue
        h, w = img.shape[:2]
        stem = os.path.splitext(os.path.basename(r.path))[0]

        vis = img.copy()

        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                xyxy = b.xyxy[0].tolist()
                confb = float(b.conf[0]) if b.conf is not None else 0.0
                cls_id = int(b.cls[0]) if b.cls is not None else 0
                cls_name = names.get(cls_id, str(cls_id))

                x1, y1, x2, y2 = map(int, xyxy)
                bbox_w = max(0, x2 - x1)
                bbox_h = max(0, y2 - y1)

                # Filtrar por tamaño mínimo: si no cumple, NO dibujar ni recortar, pero registrar en CSV
                if not is_valid_crop_size(bbox_w, bbox_h):
                    wr.writerow([
                        r.path, "", "discarded", "too_small", round(confb, 4),
                        x1, y1, x2, y2, bbox_w, bbox_h,
                        w, h, cls_id, cls_name
                    ])
                    total_discarded += 1
                    continue

                # Recorte válido
                crop = crop_xyxy(img, (x1, y1, x2, y2))
                if crop is not None:
                    crop_name = f"{stem}_{uuid.uuid4().hex[:8]}.jpg"
                    crop_path = os.path.join(crops_dir, crop_name)
                    cv2.imwrite(crop_path, crop)

                    wr.writerow([
                        r.path, crop_path, "saved", "", round(confb, 4),
                        x1, y1, x2, y2, bbox_w, bbox_h,
                        w, h, cls_id, cls_name
                    ])
                    total_saved += 1

                    # Dibujo solo para aceptados
                    draw_box(vis, (x1, y1, x2, y2), f"{cls_name} {confb:.2f}")

        # Guardar imagen anotada (aunque no haya detecciones)
        out_vis = os.path.join(ann_dir, f"{stem}_pred.jpg")
        cv2.imwrite(out_vis, vis)
        total_imgs += 1

    fcsv.close()

    # Actualizar runs/predict/latest de forma atómica y segura
    install_latest_atomic(base_dir)

    print(f"[OK] Imágenes procesadas: {total_imgs}")
    print(f"[OK] Detecciones guardadas (crops): {total_saved}")
    print(f"[OK] Detecciones descartadas por tamaño: {total_discarded}")
    print(f"[OK] Recortes en: {crops_dir}")
    print(f"[OK] Manifiesto CSV: {csv_path}")
    print(f"[OK] Imágenes anotadas en: {ann_dir}")
    print(f"[OK] Última corrida copiada a: runs/predict/latest/{{annotated,crops}}")

if __name__ == "__main__":
    main()
