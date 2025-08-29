# -*- coding: utf-8 -*-
"""
Pipeline completo con CSV detallado por detección.
- YOLOv8: detección y crops (opcional)
- SAM conmutable: --sam_mode {off,classify,save}
- Clasificación (healthy/affected)
- Salidas por corrida (timestamp) + copia runs/pipeline/latest
- CSVs:
  * csv/<stem>_analysis.csv       → por imagen
  * detections.csv                → global por detección (completo)
  * summary.csv                   → global por imagen (conteos)
  * meta.json                     → metadatos de la corrida
"""

import os, argparse, uuid, csv, glob, json, shutil, urllib.request
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO

# --- YOLO defaults (alineados con detect_and_crop_yolov8.py) ---
YOLO_DEFAULT_WEIGHTS = os.path.join("models", "best.pt")
YOLO_DEFAULT_DIR     = os.path.join("data", "yolo", "images", "test")
YOLO_DEFAULT_CONF    = 0.40
YOLO_DEFAULT_IOU     = 0.45
YOLO_DEFAULT_IMGSZ   = 960


# --------- SAM (opcional) ----------
SAM_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}
SAM_CKPT_NAME = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}

def try_import_sam():
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        return sam_model_registry, SamAutomaticMaskGenerator
    except Exception as e:
        raise ImportError(
            "No se pudo importar 'segment_anything'. Instalalo con:\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from e

# --------- Utils básicos ----------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def file_must_exist(p, what):
    if not Path(p).exists():
        raise FileNotFoundError(f"No se encontró {what}: {p}")
def clamp(v, lo, hi): return max(lo, min(hi, v))

def crop_with_pad(img, x1, y1, x2, y2, pad=0):
    h, w = img.shape[:2]
    x1 = clamp(int(x1) - pad, 0, w - 1)
    y1 = clamp(int(y1) - pad, 0, h - 1)
    x2 = clamp(int(x2) + pad, 0, w - 1)
    y2 = clamp(int(y2) + pad, 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def draw_box_label(img, xyxy, text, color, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_text = max(0, y1 - 8)
    cv2.rectangle(img, (x1, y_text - th - 4), (x1 + tw + 6, y_text + base), color, -1)
    cv2.putText(img, text, (x1 + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

# --------- Clasificador ----------
def load_classifier(weights_path, device):
    ckpt = torch.load(weights_path, map_location=device)
    classes = ckpt.get("classes", ["healthy", "affected"])
    args = ckpt.get("args", {})
    model_name = str(args.get("model", "resnet18")).lower()

    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, len(classes))
    else:
        model = models.resnet18(weights=None)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, len(classes))

    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()

    img_size = int(args.get("img_size", 384))
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return model, classes, tf

def classify_topk(model, tf, device, pil_img, k=3):
    x = tf(pil_img).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    topk = torch.topk(probs, k=min(k, probs.numel()))
    return topk  # indices, values

# --------- RGBA→RGB para clasificar ----------
def rgba_to_rgb_numpy(rgba_bgra: np.ndarray, bg: str = "black") -> np.ndarray:
    """Compone BGRA (OpenCV) a RGB con fondo negro/blanco."""
    assert rgba_bgra.shape[2] == 4
    bgr = rgba_bgra[..., :3].astype(np.float32)
    a = (rgba_bgra[..., 3:4].astype(np.float32)) / 255.0
    bg_val = 0.0 if str(bg).lower() == "black" else 255.0
    bgr_comp = bgr * a + bg_val * (1.0 - a)
    bgr_comp = np.clip(bgr_comp, 0, 255).astype('uint8')
    rgb = cv2.cvtColor(bgr_comp, cv2.COLOR_BGR2RGB)
    return rgb

# --------- SAM helpers ----------
def ckpt_path_for_model(model: str) -> Path:
    return Path("models") / SAM_CKPT_NAME[model]

def download_checkpoint_if_needed(ckpt_path: Path, model_type: str):
    if ckpt_path.exists():
        return
    url = SAM_URLS.get(model_type)
    if not url:
        raise ValueError(f"No hay URL para SAM '{model_type}'")
    ensure_dir(ckpt_path.parent)
    print(f"[SAM] Descargando checkpoint {model_type} ...")
    urllib.request.urlretrieve(url, str(ckpt_path))
    print(f"[SAM] Guardado en {ckpt_path}")

def clean_mask(mask_bool):
    m = (mask_bool > 0).astype('uint8') * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    return m

def largest_mask(masks):
    if not masks: return None
    best_idx, best_score = -1, -1.0
    for i, m in enumerate(masks):
        area = m.get("area", int(m["segmentation"].sum()))
        stab = float(m.get("stability_score", 0.0))
        score = area * (0.5 + 0.5*stab)
        if score > best_score:
            best_score, best_idx = score, i
    return masks[best_idx]["segmentation"]

def build_mask_generator(SamAutomaticMaskGenerator, sam, work_shape, min_area_frac):
    H, W = work_shape[:2]
    min_area = int(min_area_frac * (H * W))
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.90,
        stability_score_thresh=0.95,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=min_area
    )

# métricas SAM (como en tu script de cut-outs)
def mask_area_norm(m_bool: np.ndarray) -> float:
    return float(m_bool.sum()) / float(m_bool.size)

def centroid_score(m_bool: np.ndarray, W:int, H:int) -> float:
    ys, xs = np.where(m_bool)
    if len(xs) == 0: return 0.0
    cx, cy = xs.mean(), ys.mean()
    dx = abs((cx - W/2) / (W/2))
    dy = abs((cy - H/2) / (H/2))
    dist = np.hypot(dx, dy) / np.hypot(1, 1)
    return 1.0 - float(dist)

def mask_solidity(m_bool: np.ndarray) -> float:
    m = (m_bool > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 0.0
    area = sum(float(cv2.contourArea(c)) for c in cnts)
    pts = np.vstack(cnts)
    hull = cv2.convexHull(pts)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 0.0: return 0.0
    return area / hull_area

def _run_len_from_corner(m_bool: np.ndarray, corner: str):
    H, W = m_bool.shape[:2]
    if corner == 'tl':
        rx = next((x for x in range(W) if not m_bool[0,x]), W)  # run on top row
        ry = next((y for y in range(H) if not m_bool[y,0]), H)  # run on left col
        return rx, ry, bool(m_bool[0,0])
    if corner == 'tr':
        rx = next((k for k,x in enumerate(range(W-1,-1,-1)) if not m_bool[0,x]), W)
        ry = next((y for y in range(H) if not m_bool[y,W-1]), H)
        return rx, ry, bool(m_bool[0,W-1])
    if corner == 'bl':
        rx = next((x for x in range(W) if not m_bool[H-1,x]), W)
        ry = next((k for k,y in enumerate(range(H-1,-1,-1)) if not m_bool[y,0]), H)
        return rx, ry, bool(m_bool[H-1,0])
    if corner == 'br':
        rx = next((k for k,x in enumerate(range(W-1,-1,-1)) if not m_bool[H-1,x]), W)
        ry = next((k for k,y in enumerate(range(H-1,-1,-1)) if not m_bool[y,W-1]), H)
        return rx, ry, bool(m_bool[H-1,W-1])
    return 0,0,False

def is_corner_wedge(mask_bool: np.ndarray, frac_thresh: float = 0.50) -> bool:
    H, W = mask_bool.shape[:2]
    if H == 0 or W == 0: return False
    for c in ('tl','tr','bl','br'):
        run_x, run_y, on = _run_len_from_corner(mask_bool, c)
        if on and run_x >= frac_thresh*W and run_y >= frac_thresh*H:
            return True
    return False

def bbox_from_mask(mask_bool: np.ndarray):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0: return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def segment_crop_with_sam_metrics(sam_modules, sam, bgr_crop,
                                  max_width=256, min_area_frac=0.20, coverage_full=0.88,
                                  corner_frac=0.50):
    """Devuelve (rgba|None, metrics_dict|None). Métricas refieren al tamaño del crop."""
    _, SamAutomaticMaskGenerator = sam_modules
    H0, W0 = bgr_crop.shape[:2]
    if W0 > max_width:
        s = max_width / float(W0)
        work = cv2.resize(bgr_crop, (int(W0*s), int(H0*s)), interpolation=cv2.INTER_AREA)
    else:
        work = bgr_crop

    work_rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    mask_gen = build_mask_generator(SamAutomaticMaskGenerator, sam, work.shape, min_area_frac)

    with torch.inference_mode():
        masks = mask_gen.generate(work_rgb)

    if len(masks) == 0:
        return None, None

    m_work = largest_mask(masks)
    if m_work is None:
        return None, None

    m_orig_bool = cv2.resize(m_work.astype('uint8'), (W0, H0), interpolation=cv2.INTER_NEAREST) > 0
    m_orig_255 = clean_mask(m_orig_bool)
    m_orig_bool = (m_orig_255 > 0)

    cov = mask_area_norm(m_orig_bool)
    sol = mask_solidity(m_orig_bool)
    ctr = centroid_score(m_orig_bool, W0, H0)
    wedge = is_corner_wedge(m_orig_bool, frac_thresh=corner_frac)

    # full coverage → usar crop completo con alpha 255
    if cov >= coverage_full:
        rgba = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2BGRA)
        rgba[..., 3] = 255
        bb = (0,0,W0-1,H0-1)
        metrics = dict(coverage=cov, solidity=sol, center_score=ctr, corner_wedge=wedge,
                       sam_x1=bb[0], sam_y1=bb[1], sam_x2=bb[2], sam_y2=bb[3],
                       sam_w=W0, sam_h=H0, sam_area=W0*H0, sam_area_rel=1.0)
        return rgba, metrics

    bb = bbox_from_mask(m_orig_bool)
    if bb is None:
        return None, None
    x1, y1, x2, y2 = bb
    crop_bgr = bgr_crop[y1:y2+1, x1:x2+1]
    crop_msk = (m_orig_bool[y1:y2+1, x1:x2+1] * 255).astype(np.uint8)
    rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = crop_msk

    sam_w = int(x2-x1+1); sam_h = int(y2-y1+1)
    sam_area = sam_w * sam_h
    metrics = dict(coverage=cov, solidity=sol, center_score=ctr, corner_wedge=wedge,
                   sam_x1=x1, sam_y1=y1, sam_x2=x2, sam_y2=y2,
                   sam_w=sam_w, sam_h=sam_h, sam_area=sam_area,
                   sam_area_rel=float(sam_area)/(W0*H0))
    return rgba, metrics

# --------- Latest helpers ----------
def safe_copytree(src: str, dst: str) -> None:
    ensure_dir(dst)
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target_root = os.path.join(dst, rel) if rel != "." else dst
        ensure_dir(target_root)
        for d in dirs:
            ensure_dir(os.path.join(target_root, d))
        for f in files:
            shutil.copy2(os.path.join(root, f), os.path.join(target_root, f))

def install_latest_atomic(run_root: str) -> None:
    parent = os.path.join("runs", "pipeline")
    ensure_dir(parent)
    tmp = os.path.join(parent, ".latest_tmp")
    latest = os.path.join(parent, "latest")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    elif os.path.islink(tmp) or os.path.isfile(tmp):
        try: os.unlink(tmp)
        except Exception: pass
    ensure_dir(tmp)
    safe_copytree(run_root, tmp)
    if os.path.islink(latest) or os.path.isfile(latest):
        try: os.unlink(latest)
        except Exception: pass
    elif os.path.isdir(latest):
        shutil.rmtree(latest, ignore_errors=True)
    os.replace(tmp, latest)

# --------- Proceso por imagen ----------
def process_one_image(image_path, yolo, clf, cls_names, clf_tf, device, args,
                      dirs, wr_summary, wr_det, run_id, sam_ctx=None):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[WARN] No se pudo leer: {image_path}")
        return 0,0,0
    H, W = img_bgr.shape[:2]
    stem = Path(image_path).stem

    results = yolo.predict(
        source=image_path,
        imgsz=args.yolo_imgsz,
        conf=args.yolo_conf,
        iou=args.yolo_iou,
        device=args.yolo_device,
        verbose=False
    )
    r = next(iter(results))
    boxes = r.boxes

    annotated = img_bgr.copy()
    total, healthy_cnt, affected_cnt = 0,0,0

    # CSV por imagen
    csv_img_path = dirs['csv'] / f"{stem}_analysis.csv"
    with open(csv_img_path, "w", newline="", encoding="utf-8") as fcsv:
        wr = csv.writer(fcsv)
        # Encabezado por imagen (simple)
        wr.writerow([
            "orig_file","crop_file","sam_file","det_conf",
            "x1","y1","x2","y2","class","cls_score","img_w","img_h"
        ])

        if boxes is not None and len(boxes)>0:
            for b in boxes:
                xyxy = b.xyxy[0].tolist()
                det_conf = float(b.conf[0]) if b.conf is not None else 0.0
                x1,y1,x2,y2 = xyxy
                bw, bh = int(x2-x1), int(y2-y1)
                barea = max(bw,0) * max(bh,0)
                barea_rel = float(barea) / float(W*H) if W*H>0 else 0.0

                crop = crop_with_pad(img_bgr, x1,y1,x2,y2, pad=args.pad)
                if crop is None: 
                    continue
                cH, cW = crop.shape[:2]

                sam_path_str = ""
                rgba_for_clf = None
                sam_used = False
                sam_metrics = dict(coverage="", solidity="", center_score="", corner_wedge="",
                                   sam_x1="", sam_y1="", sam_x2="", sam_y2="",
                                   sam_w="", sam_h="", sam_area="", sam_area_rel="")
                if args.sam_mode in ("classify","save") and sam_ctx is not None:
                    rgba, metrics = segment_crop_with_sam_metrics(
                        sam_ctx['modules'], sam_ctx['sam'], crop,
                        max_width=args.sam_max_width,
                        min_area_frac=args.sam_min_area_frac,
                        coverage_full=args.sam_coverage_full,
                        corner_frac=args.sam_corner_frac
                    )
                    if rgba is not None:
                        rgba_for_clf = rgba
                        sam_used = True
                        sam_metrics = metrics

                # Clasificar
                if rgba_for_clf is not None:
                    rgb_np = rgba_to_rgb_numpy(rgba_for_clf, bg=args.sam_bg)
                    pil_input = Image.fromarray(rgb_np)
                else:
                    pil_input = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

                topk = classify_topk(clf, clf_tf, device, pil_input, k=args.clf_topk)
                pred_idx = int(topk.indices[0].item())
                pred_name = cls_names[pred_idx]
                pred_score = float(topk.values[0].item())
                topk_names = [cls_names[int(i.item())] for i in topk.indices]
                topk_scores = [float(v.item()) for v in topk.values]

                total += 1
                if pred_name.lower().startswith("heal"):
                    healthy_cnt += 1; color=(0,200,0)
                else:
                    affected_cnt += 1; color=(0,0,255)

                # Guardar crop YOLO (opcional)
                crop_path_str = ""
                if args.save_crops:
                    cls_folder = "healthy" if pred_name.lower().startswith("heal") else "affected"
                    out_dir_crops = (dirs['crops'] / cls_folder) if args.separate_by_class else dirs['crops']
                    ensure_dir(out_dir_crops)
                    crop_name = f"{stem}_{uuid.uuid4().hex[:8]}_yolocrop.jpg"
                    crop_path_str = str(out_dir_crops / crop_name)
                    cv2.imwrite(crop_path_str, crop)

                # Guardar SAM RGBA si corresponde
                if (args.sam_mode == "save") and (rgba_for_clf is not None):
                    cls_folder = "healthy" if pred_name.lower().startswith("heal") else "affected"
                    out_dir_sam = (dirs['sam'] / cls_folder) if args.separate_by_class else dirs['sam']
                    ensure_dir(out_dir_sam)
                    sam_name = f"{stem}_{uuid.uuid4().hex[:8]}_sam.png"
                    sam_path_str = str(out_dir_sam / sam_name)
                    cv2.imwrite(sam_path_str, rgba_for_clf)

                # Anotar en imagen
                label = f"{pred_name} {pred_score:.2f} | det {det_conf:.2f}"
                draw_box_label(annotated, (x1,y1,x2,y2), label, color)

                # CSV por imagen (simple)
                wr.writerow([image_path, crop_path_str, sam_path_str, f"{det_conf:.4f}",
                             int(x1),int(y1),int(x2),int(y2), pred_name, f"{pred_score:.4f}", W,H])

                # CSV global por detección (completo)
                row_det = [
                    run_id,
                    Path(image_path).name, W, H,
                    f"{det_conf:.4f}",
                    int(x1), int(y1), int(x2), int(y2),
                    bw, bh, barea, f"{barea_rel:.6f}",
                    crop_path_str,
                    # Clasificación
                    pred_name, f"{pred_score:.4f}",
                ]
                # TopK dinámico
                for n in topk_names:
                    row_det.append(n)
                for s in topk_scores:
                    row_det.append(f"{s:.4f}")
                # SAM métricas
                row_det.extend([
                    int(sam_used),
                    sam_path_str,
                    "" if sam_metrics["coverage"]=="" else f"{sam_metrics['coverage']:.6f}",
                    "" if sam_metrics["solidity"]=="" else f"{sam_metrics['solidity']:.6f}",
                    "" if sam_metrics["center_score"]=="" else f"{sam_metrics['center_score']:.6f}",
                    "" if sam_metrics["corner_wedge"]=="" else int(bool(sam_metrics["corner_wedge"])),
                    # BBox SAM dentro del crop
                    sam_metrics["sam_x1"], sam_metrics["sam_y1"],
                    sam_metrics["sam_x2"], sam_metrics["sam_y2"],
                    sam_metrics["sam_w"], sam_metrics["sam_h"],
                    sam_metrics["sam_area"],
                    "" if sam_metrics["sam_area_rel"]=="" else f"{sam_metrics['sam_area_rel']:.6f}",
                    # tamaños de crop (referencia para áreas relativas)
                    cW, cH
                ])
                wr_det.writerow(row_det)

    out_vis = dirs['predict'] / f"{stem}_analysis.jpg"
    cv2.imwrite(str(out_vis), annotated)

    pct_h = (healthy_cnt/total*100.0) if total>0 else 0.0
    pct_a = (affected_cnt/total*100.0) if total>0 else 0.0
    wr_summary.writerow([Path(image_path).name, total, healthy_cnt, affected_cnt, f"{pct_h:.2f}", f"{pct_a:.2f}"])

    print(f"[{stem}] hojas={total}, healthy={healthy_cnt}, affected={affected_cnt}")
    return total, healthy_cnt, affected_cnt

# --------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--image", help="Ruta a una sola imagen")
    # ap.add_argument("--dir", help="Ruta a una carpeta de imágenes")

    # Entradas
    ap.add_argument("--image", default=None, help="Ruta a una sola imagen")
    ap.add_argument("--dir", default=YOLO_DEFAULT_DIR, help="Ruta a una carpeta de imágenes")


    # YOLO
    ap.add_argument("--yolo_weights", default=YOLO_DEFAULT_WEIGHTS, help="Pesos YOLOv8 (def: models/best.pt)")
    ap.add_argument("--yolo_imgsz", type=int,   default=YOLO_DEFAULT_IMGSZ)
    ap.add_argument("--yolo_conf",  type=float, default=YOLO_DEFAULT_CONF)
    ap.add_argument("--yolo_iou",   type=float, default=YOLO_DEFAULT_IOU)
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--save_crops", action="store_true", help="Guarda recortes de YOLO")
    ap.add_argument("--separate_by_class", action="store_true", help="Guarda crops/SAM en subcarpetas healthy/affected")

    ap.add_argument("--clf_weights", default=os.path.join("models","resnet18_best.pt"),
                    help="Pesos clasificador (def: models/resnet18_best.pt)")

    # Salidas
    ap.add_argument("--out_dir", default=os.path.join("runs","pipeline"),
                    help="Carpeta raíz de salidas")
    ap.add_argument("--run_name", default=None, help="Nombre opcional para prefijo de la corrida")
   
    # Clasificación
    ap.add_argument("--clf_topk", type=int, default=3, help="Top-K a registrar en CSV")

    # --- SAM conmutable
    ap.add_argument("--sam_mode", default="off", choices=["off","classify","save"],
                    help="off: deshabilitado | classify: usa SAM para clasificar | save: usa y guarda RGBA")
    # 👉 por defecto ahora vit_h (igual que en sam_segment_crops.py)
    ap.add_argument("--sam_model", default="vit_h", choices=["vit_b","vit_l","vit_h"])
    ap.add_argument("--sam_checkpoint", default=None,
                    help="Ruta al checkpoint SAM (.pth). Si no se pasa, usa models/<oficial>.pth")
    # 👉 alineamos defaults “estilo SAM crops”
    #    - min_area_frac: 0.10 (como el preset ultra_recall_crops)
    #    - coverage_full : 0.95 (mismo “piso” efectivo del script de cut-outs)
    ap.add_argument("--sam_max_width", type=int, default=256)
    ap.add_argument("--sam_min_area_frac", type=float, default=0.10)
    ap.add_argument("--sam_coverage_full", type=float, default=0.95,
                    help="Si la máscara cubre >= este valor, el PNG es el crop completo con alpha=255")
    ap.add_argument("--sam_bg", type=str, default="black", choices=["black","white"],
                    help="Fondo para componer RGBA→RGB al clasificar")

    # 👉 corner_frac por defecto 0.60 (como pediste en sam_segment_crops.py)
    ap.add_argument("--sam_corner_frac", type=float, default=0.60,
                    help="Umbral para marcar 'corner_wedge' (detección de cuña en esquina)")

    # Compat antiguo
    ap.add_argument("--save_sam", action="store_true", help="[DEPRECADO] Equivalente a --sam_mode save")

    args = ap.parse_args()
    if args.save_sam:
        args.sam_mode = "save"

    # Checks y armado de salidas con timestamp
    file_must_exist(args.yolo_weights, "pesos YOLO")
    file_must_exist(args.clf_weights, "pesos clasificador")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_tag = f"{args.run_name}_" if args.run_name else ""
    run_id = f"{run_tag}{stamp}"

    out_root = Path(args.out_dir) / run_id
    dirs = {
        'root': out_root,
        'predict': out_root / "predict",
        'crops':   out_root / "crops",
        'sam':     out_root / "sam",
        'csv':     out_root / "csv",
    }
    for d in dirs.values(): ensure_dir(d)

    if args.separate_by_class:
        if args.save_crops:
            ensure_dir(dirs['crops'] / "healthy"); ensure_dir(dirs['crops'] / "affected")
        if args.sam_mode == "save":
            ensure_dir(dirs['sam'] / "healthy"); ensure_dir(dirs['sam'] / "affected")


    # Device estilo detect_and_crop_yolov8.py
    yolo_device = "0" if torch.cuda.is_available() else "cpu"
    setattr(args, "yolo_device", yolo_device)

    # 👉 agregar esto:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] YOLO device: {args.yolo_device} | CLF/SAM device: {device}")

    # Modelos
    yolo = YOLO(args.yolo_weights)
    clf, cls_names, clf_tf = load_classifier(args.clf_weights, device)


    # SAM (si corresponde)
    sam_ctx = None
    ckpt_path_used = None
    if args.sam_mode in ("classify","save"):
        ckpt_path_used = Path(args.sam_checkpoint) if args.sam_checkpoint else ckpt_path_for_model(args.sam_model)
        download_checkpoint_if_needed(ckpt_path_used, args.sam_model)
        sam_model_registry, SamAutomaticMaskGenerator = try_import_sam()
        print(f"[SAM] Cargando {args.sam_model} desde {ckpt_path_used} ...")
        sam = sam_model_registry[args.sam_model](checkpoint=str(ckpt_path_used)).to(device)
        sam_ctx = {'modules': (sam_model_registry, SamAutomaticMaskGenerator), 'sam': sam}

    # CSVs globales
    summary_path = dirs['root'] / "summary.csv"
    det_csv_path = dirs['root'] / "detections.csv"
    meta_path    = dirs['root'] / "meta.json"

    # Header dinámico para detections.csv con Top-K
    det_header = [
        "run_id",
        "file","img_w","img_h",
        "yolo_conf",
        "x1","y1","x2","y2","bbox_w","bbox_h","bbox_area","bbox_area_rel",
        "crop_file",
        "pred_class","pred_score",
    ]
    det_header += [f"top{i}_class" for i in range(1, args.clf_topk+1)]
    det_header += [f"top{i}_score" for i in range(1, args.clf_topk+1)]
    det_header += [
        "sam_used","sam_file",
        "sam_coverage","sam_solidity","sam_center_score","sam_corner_wedge",
        "sam_x1","sam_y1","sam_x2","sam_y2","sam_w","sam_h","sam_area","sam_area_rel",
        "crop_w","crop_h"
    ]

    with open(summary_path, "w", newline="", encoding="utf-8") as fs, \
         open(det_csv_path, "w", newline="", encoding="utf-8") as fd:
        wr_sum = csv.writer(fs); wr_det = csv.writer(fd)
        wr_sum.writerow(["file","total","healthy","affected","pct_healthy","pct_affected"])
        wr_det.writerow(det_header)

        total_all, healthy_all, affected_all = 0,0,0

        # Imagen única
        if args.image:
            file_must_exist(args.image, "imagen de entrada")
            t,h,a = process_one_image(args.image, yolo, clf, cls_names, clf_tf, device,
                                      args, dirs, wr_sum, wr_det, run_id, sam_ctx)
            total_all+=t; healthy_all+=h; affected_all+=a

        # Carpeta completa
        if args.dir:
            exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")
            imgs = [p for p in glob.glob(os.path.join(args.dir,"*")) if p.lower().endswith(exts)]
            for p in imgs:
                t,h,a = process_one_image(p, yolo, clf, cls_names, clf_tf, device,
                                          args, dirs, wr_sum, wr_det, run_id, sam_ctx)
                total_all+=t; healthy_all+=h; affected_all+=a

        pct_h = (healthy_all/total_all*100.0) if total_all>0 else 0.0
        pct_a = (affected_all/total_all*100.0) if total_all>0 else 0.0
        wr_sum.writerow(["TOTAL", total_all, healthy_all, affected_all, f"{pct_h:.2f}", f"{pct_a:.2f}"])

    # meta corrida
    meta = {
        "run_id": run_id,
        "timestamp": run_id.split("_")[-1],
        "run_dir": str(dirs['root']),
        "yolo_weights": args.yolo_weights,
        "clf_weights": args.clf_weights,
        "classes": list(map(str, cls_names)),
        "sam_mode": args.sam_mode,
        "sam_model": args.sam_model if args.sam_mode!="off" else None,
        "sam_checkpoint": str(ckpt_path_used) if ckpt_path_used else None,
        "yolo_imgsz": args.yolo_imgsz,
        "yolo_conf": args.yolo_conf,
        "yolo_iou": args.yolo_iou,
        "yolo_device": args.yolo_device,
        "pad": args.pad,
        "save_crops": bool(args.save_crops),
        "separate_by_class": bool(args.separate_by_class),
        "sam_max_width": args.sam_max_width,
        "sam_min_area_frac": args.sam_min_area_frac,
        "sam_coverage_full": args.sam_coverage_full,
        "sam_bg": args.sam_bg,
        "sam_corner_frac": args.sam_corner_frac,
        "clf_topk": args.clf_topk,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # instalar latest
    install_latest_atomic(str(dirs['root']))

    print("\n[RESUMEN GLOBAL]")
    print(f"Summary   : {summary_path}")
    print(f"Detections: {det_csv_path}")
    print(f"Meta      : {meta_path}")
    print(f"Salida    : {dirs['root']}")
    print(f"Latest    : {Path('runs/pipeline/latest').resolve()}")

if __name__ == "__main__":
    main()


# SAM deshabilitado (solo YOLO + clasificación):
# python scripts/analyze_image.py --dir data/yolo/images/test --yolo_weights models/best.pt --clf_weights models/resnet18_best.pt --clf_topk 3

# SAM para clasificar (no guarda PNG), fondo negro al componer:
# python scripts/analyze_image.py --dir data/yolo/images/test --yolo_weights models/best.pt --clf_weights models/resnet18_best.pt --sam_mode classify --sam_model vit_h --sam_bg black --sam_max_width 256 --sam_min_area_frac 0.20 --sam_coverage_full 0.88 --clf_topk 3

# SAM guardando PNG RGBA + separar por clase:
# python scripts/analyze_image.py --dir data/yolo/images/test --yolo_weights models/best.pt --clf_weights models/resnet18_best.pt --sam_mode save --sam_model vit_h --separate_by_class --save_crops --clf_topk 3
