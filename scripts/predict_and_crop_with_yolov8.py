# -*- coding: utf-8 -*-
"""
Predice hojas con YOLOv8 y recorta cada detección a disco.
- Lee imágenes de --source (carpeta con .jpg/.png)
- Usa --weights (best.pt entrenado)
- Guarda recortes en --out_dir y un CSV con metadatos de cada recorte
"""
import argparse, os, csv, uuid, cv2
from ultralytics import YOLO

def ensure_dir(d): os.makedirs(d, exist_ok=True)

def crop_xyxy(img, xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="runs/detect/train/weights/best.pt")
    ap.add_argument("--source",  required=True, help="Carpeta de imágenes a procesar")
    ap.add_argument("--out_dir", required=True, help="Carpeta de salida de recortes")
    ap.add_argument("--conf", type=float, default=0.25, help="Confianza mínima")
    ap.add_argument("--iou",  type=float, default=0.45, help="IoU para NMS")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    csv_path = os.path.join(args.out_dir, "crops_manifest.csv")
    fcsv = open(csv_path, "w", newline="", encoding="utf-8")
    wr = csv.writer(fcsv)
    wr.writerow(["orig_file","crop_file","conf","x1","y1","x2","y2","w","h"])

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        stream=True,
        verbose=True
    )

    for r in results:
        img = r.orig_img
        if img is None: 
            continue
        h, w = img.shape[:2]
        stem = os.path.splitext(os.path.basename(r.path))[0]

        if r.boxes is None: 
            continue
        for b in r.boxes:
            xyxy = b.xyxy[0].tolist()
            conf = float(b.conf[0]) if b.conf is not None else 0.0
            crop = crop_xyxy(img, xyxy)
            if crop is None: 
                continue

            crop_name = f"{stem}_{uuid.uuid4().hex[:8]}.jpg"
            crop_path = os.path.join(args.out_dir, crop_name)
            cv2.imwrite(crop_path, crop)

            x1, y1, x2, y2 = map(int, xyxy)
            wr.writerow([r.path, crop_path, round(conf,4), x1, y1, x2, y2, w, h])

    fcsv.close()
    print(f"[OK] Recortes en: {args.out_dir}")
    print(f"[OK] Manifiesto CSV: {csv_path}")

if __name__ == "__main__":
    main()
