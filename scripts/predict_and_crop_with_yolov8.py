# -*- coding: utf-8 -*-
"""
Predice hojas con YOLOv8, recorta cada detección a disco y guarda una imagen anotada por input.
- Lee imágenes de --source (carpeta con .jpg/.png)
- Usa --weights (best.pt entrenado)
- Guarda recortes en --out_dir y un CSV con metadatos de cada recorte
- Guarda imágenes anotadas (bboxes + confidencia) en --pred_dir (por default: carpeta 'predict' junto a --out_dir)
"""
import argparse, os, csv, uuid, cv2
from ultralytics import YOLO

def ensure_dir(d): os.makedirs(d, exist_ok=True)

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
    # Caja del texto
    (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_text = max(0, y1 - 8)
    cv2.rectangle(img, (x1, y_text - th - 4), (x1 + tw + 6, y_text + baseline), color, -1)
    cv2.putText(img, label_text, (x1 + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="runs/detect/train/weights/best.pt")
    ap.add_argument("--source",  required=True, help="Carpeta de imágenes a procesar")
    ap.add_argument("--out_dir", required=True, help="Carpeta de salida de recortes (p.ej. data/crops)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confianza mínima")
    ap.add_argument("--iou",  type=float, default=0.45, help="IoU para NMS")
    ap.add_argument("--imgsz", type=int, default=640, help="Tamaño de entrada para YOLO")
    ap.add_argument("--pred_dir", default=None,
                    help="Carpeta para imágenes anotadas. Si no se indica, crea 'predict' junto a --out_dir.")
    args = ap.parse_args()

    # Salidas
    ensure_dir(args.out_dir)
    pred_dir = args.pred_dir
    if pred_dir is None:
        parent = os.path.dirname(os.path.abspath(args.out_dir))
        pred_dir = os.path.join(parent, "predict")
    ensure_dir(pred_dir)

    # CSV
    csv_path = os.path.join(args.out_dir, "crops_manifest.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["orig_file", "crop_file", "conf", "x1", "y1", "x2", "y2", "img_w", "img_h"])

        # Modelo
        model = YOLO(args.weights)
        names = getattr(model, "names", None)
        if not names:
            # fallback: clase 0 = 'leaf'
            names = {0: "leaf"}

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

            # Copia para anotaciones
            vis = img.copy()
            any_det = False

            if r.boxes is None:
                # Guardar igualmente una imagen (sin detecciones) si se desea
                out_vis = os.path.join(pred_dir, f"{stem}_pred.jpg")
                cv2.imwrite(out_vis, vis)
                continue

            for b in r.boxes:
                xyxy = b.xyxy[0].tolist()
                conf = float(b.conf[0]) if b.conf is not None else 0.0
                cls_id = int(b.cls[0]) if b.cls is not None else 0
                cls_name = names.get(cls_id, str(cls_id))

                # Recorte
                crop = crop_xyxy(img, xyxy)
                if crop is not None:
                    crop_name = f"{stem}_{uuid.uuid4().hex[:8]}.jpg"
                    crop_path = os.path.join(args.out_dir, crop_name)
                    cv2.imwrite(crop_path, crop)

                    x1, y1, x2, y2 = map(int, xyxy)
                    wr.writerow([r.path, crop_path, round(conf, 4), x1, y1, x2, y2, w, h])

                # Dibujo en la imagen
                label_text = f"{cls_name} {conf:.2f}"
                draw_box(vis, xyxy, label_text)
                any_det = True

            # Guardar imagen anotada (aunque no haya detecciones, para trazabilidad)
            out_vis = os.path.join(pred_dir, f"{stem}_pred.jpg")
            cv2.imwrite(out_vis, vis)

    print(f"[OK] Recortes en: {args.out_dir}")
    print(f"[OK] Manifiesto CSV: {csv_path}")
    print(f"[OK] Imágenes anotadas en: {pred_dir}")

if __name__ == "__main__":
    main()
