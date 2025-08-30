# -*- coding: utf-8 -*-
"""
Anota las imágenes originales con 'healthy' / 'affected' usando:
- runs/predict/latest/crops/crops_manifest.csv   (bbox por crop)
- runs/classify/predict/latest/predictions.csv   (clase por cut-out SAM)

Salida:
- runs/predict/latest/annotated_cls/<stem>_cls.jpg
"""

import os, csv
from pathlib import Path
import cv2

MANIFEST = Path("runs/predict/latest/crops/crops_manifest.csv")
PREDCSV  = Path("runs/classify/predict/latest/predictions.csv")
OUTDIR   = Path("runs/predict/latest/annotated_cls")

def color_for(label: str):
    return (0,200,0) if label.lower().startswith("heal") else (0,0,255)

def draw_box(img, xyxy, text, color, thickness=2):
    x1,y1,x2,y2 = map(int, xyxy)
    cv2.rectangle(img,(x1,y1),(x2,y2),color,thickness)
    (tw,th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_text = max(0, y1-8)
    cv2.rectangle(img,(x1, y_text-th-4),(x1+tw+6, y_text+base), color, -1)
    cv2.putText(img, text, (x1+3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

def best_preds_by_cropstem(pred_csv: Path):
    """Devuelve {crop_stem: (label, score)} tomando el mejor score si hay varios cut-outs."""
    by_crop = {}
    if not pred_csv.is_file():
        print(f"[WARN] No existe {pred_csv}")
        return by_crop
    with open(pred_csv, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            rgba_path = Path(row["file"])
            stem = rgba_path.stem  # p.ej: IMG_0001_abcd1234_yolocrop_cut01
            # recortar sufijo _cutNN → stem del crop YOLO
            cut_pos = stem.rfind("_cut")
            crop_stem = stem[:cut_pos] if cut_pos != -1 else stem
            label = row["pred_class"]
            try:
                score = float(row["pred_score"])
            except:
                score = 0.0
            if (crop_stem not in by_crop) or (score > by_crop[crop_stem][1]):
                by_crop[crop_stem] = (label, score)
    return by_crop

def main():
    if not MANIFEST.is_file():
        print(f"[ERR] No se encuentra {MANIFEST}")
        return
    preds = best_preds_by_cropstem(PREDCSV)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Agrupar por imagen original
    per_orig = {}  # {orig_path: [ (x1,y1,x2,y2,label,score) ]}
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            if row.get("status","") != "saved":  # solo crops aceptados
                continue
            crop_path = Path(row["crop_file"])
            crop_stem = crop_path.stem
            if crop_stem not in preds:
                continue
            label, score = preds[crop_stem]
            x1,y1,x2,y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
            orig = Path(row["orig_file"])
            per_orig.setdefault(orig, []).append((x1,y1,x2,y2,label,score))

    # Dibujar por imagen original
    for orig, items in per_orig.items():
        img = cv2.imread(str(orig))
        if img is None:
            print(f"[WARN] No se pudo leer {orig}")
            continue
        for (x1,y1,x2,y2,label,score) in items:
            draw_box(img, (x1,y1,x2,y2), f"{label} {score:.2f}", color_for(label))
        out_path = OUTDIR / (orig.stem + "_cls.jpg")
        cv2.imwrite(str(out_path), img)
        print(f"[OK] {out_path}")

if __name__ == "__main__":
    main()
