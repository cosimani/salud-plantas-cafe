# scripts/sam_cutouts.py
# -*- coding: utf-8 -*-
"""
Cut-Outs sobre CROPS de YOLO con backend conmutable:
- BACKEND="sam"  -> SAM AutomaticMaskGenerator (default)
- BACKEND="rembg"-> Rembg (ISNet general)
- Fallback: si SAM no produce máscaras >= AREA_RATIO_THRESH, intenta Rembg.

Entrada: runs/predict/latest/crops/
Salida:  runs/segment_cutouts/<timestamp>/rgba/  +  runs/segment_cutouts/latest/rgba/
"""

import os, cv2, glob, shutil, urllib.request, numpy as np, torch, io
from datetime import datetime
from typing import List, Dict
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ---------------- Config ----------------
IN_DIR = os.path.join("runs", "predict", "latest", "crops")
OUT_PARENT = os.path.join("runs", "segment_cutouts")
OUT_SUB    = "rgba"

# Backend: "sam" (default) | "rembg"
BACKEND = "sam"
USE_REMBG_FALLBACK_IF_SAM_FAILS = True

# SAM
MODEL_TYPE = "vit_b"
CKPT_PATH  = os.path.join("models", "sam_vit_b_01ec64.pth")
SAM_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}

# Ajustes SAM para crops
POINTS_PER_SIDE   = 16
PRED_IOU_THRESH   = 0.88
STAB_SCORE_THRESH = 0.92
MIN_AREA_FRAC     = 0.03
MAX_CUTS_PER_IMG  = 4
MAX_WIDTH         = 512
IOU_SUPPRESS      = 0.55
CENTER_BONUS      = 0.20

# Filtro fuerte: cobertura mínima del 70% del crop
AREA_RATIO_THRESH = 0.30

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Rembg (se importa on-demand)
try:
    from rembg import remove, new_session
    from PIL import Image
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False

# ---------------- Utils ----------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def list_images(path: str) -> List[str]:
    files = []
    for ext in EXTS:
        files.extend(glob.glob(os.path.join(path, f"*{ext}")))
        files.extend(glob.glob(os.path.join(path, f"*{ext.upper()}")))
    return sorted(files)

def download_ckpt_if_needed(path: str, model_type: str):
    if os.path.exists(path): return
    url = SAM_URLS.get(model_type)
    if not url: raise ValueError(f"No hay URL para modelo {model_type}")
    ensure_dir(os.path.dirname(path))
    print(f"[INFO] Descargando SAM {model_type} desde {url} ...")
    urllib.request.urlretrieve(url, path)
    print(f"[OK] Guardado en {path}")

def build_mask_generator(sam, H: int, W: int):
    min_area = int(MIN_AREA_FRAC * (H * W))
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        stability_score_offset=1.0,
        box_nms_thresh=0.7,
        crop_n_layers=0,
        crop_nms_thresh=0.7,
        crop_overlap_ratio=512/1500,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=0,
        output_mode="binary_mask"
    )

def mask_area(m: Dict) -> int:
    return int(m.get("area", int(m["segmentation"].sum())))

def mask_centroid(seg_bool: np.ndarray):
    ys, xs = np.where(seg_bool > 0)
    if len(xs) == 0: return None
    return (float(xs.mean()), float(ys.mean()))

def score_centered(seg_bool: np.ndarray, W: int, H: int) -> float:
    c = mask_centroid(seg_bool)
    if c is None: return 0.0
    dx = abs((c[0] - W/2) / (W/2))
    dy = abs((c[1] - H/2) / (H/2))
    dist = np.hypot(dx, dy) / np.hypot(1, 1)
    return 1.0 - float(dist)

def iou_masks(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 0.0 if union == 0 else float(inter) / float(union)

def nms_by_iou(masks: List[np.ndarray], scores: List[float], iou_thr: float):
    idxs = np.argsort(scores)[::-1]
    keep, used = [], np.zeros(len(masks), dtype=bool)
    for i in idxs:
        if used[i]: continue
        keep.append(i); used[i] = True
        for j in idxs:
            if used[j]: continue
            if iou_masks(masks[i], masks[j]) >= iou_thr:
                used[j] = True
    return keep

def to_rgba_crop(img_bgr: np.ndarray, mask_bool: np.ndarray):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0: return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    crop_bgr = img_bgr[y1:y2+1, x1:x2+1]
    crop_msk = (mask_bool[y1:y2+1, x1:x2+1] * 255).astype(np.uint8)
    rgba = np.zeros((crop_bgr.shape[0], crop_bgr.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = crop_bgr
    rgba[..., 3] = crop_msk
    return rgba

def safe_copytree(src: str, dst: str) -> None:
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

def install_latest_atomic(run_root: str):
    ensure_dir("runs"); ensure_dir(OUT_PARENT)
    tmp = os.path.join(OUT_PARENT, ".latest_tmp")
    latest = os.path.join(OUT_PARENT, "latest")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    elif os.path.islink(tmp) or os.path.isfile(tmp):
        try: os.unlink(tmp)
        except Exception: pass
    ensure_dir(tmp)
    src_sub = os.path.join(run_root, OUT_SUB)
    if os.path.isdir(src_sub):
        dst_sub = os.path.join(tmp, OUT_SUB)
        safe_copytree(src_sub, dst_sub)
    if os.path.islink(latest) or os.path.isfile(latest):
        try: os.unlink(latest)
        except Exception: pass
    elif os.path.isdir(latest):
        shutil.rmtree(latest, ignore_errors=True)
    os.replace(tmp, latest)

# ---------------- Rembg helpers ----------------
REMBG_SESSION = None
def get_rembg_session():
    global REMBG_SESSION
    if REMBG_SESSION is None:
        if not REMBG_AVAILABLE:
            raise RuntimeError("rembg no está instalado. Ejecutá: pip install rembg pillow")
        REMBG_SESSION = new_session("isnet-general-use")
    return REMBG_SESSION

def rembg_cutout(img_bgr: np.ndarray):
    """Devuelve (rgba_crop, coverage_ratio) o (None, 0.0)."""
    if not REMBG_AVAILABLE:
        return None, 0.0
    from PIL import Image
    session = get_rembg_session()
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO(); pil.save(buf, format="PNG"); buf.seek(0)
    out_bytes = remove(buf.getvalue(), session=session)
    if not out_bytes:
        return None, 0.0
    res = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
    rgba = np.array(res)
    alpha = rgba[..., 3] > 0
    if alpha.sum() == 0:
        return None, 0.0
    # recorte al bbox del alfa
    ys, xs = np.where(alpha)
    x1, x2 = xs.min(), xs.max(); y1, y2 = ys.min(), ys.max()
    crop_rgba = rgba[y1:y2+1, x1:x2+1, :]
    coverage = float(alpha.sum()) / float(alpha.size)
    return crop_rgba, coverage

# ---------------- Main ----------------
def main():
    if not os.path.isdir(IN_DIR):
        print(f"[ERROR] No se encontró {IN_DIR}. Ejecutá primero detect_and_crop_yolov8.py.")
        return

    ensure_dir("runs"); ensure_dir(OUT_PARENT)
    run_id  = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = os.path.join(OUT_PARENT, run_id)
    out_dir  = os.path.join(out_root, OUT_SUB)
    ensure_dir(out_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Backend: {BACKEND}")

    if BACKEND == "sam":
        download_ckpt_if_needed(CKPT_PATH, MODEL_TYPE)
        sam = sam_model_registry[MODEL_TYPE](checkpoint=CKPT_PATH).to(device)
    elif BACKEND == "rembg":
        if not REMBG_AVAILABLE:
            raise SystemExit("rembg no está instalado. Ejecutá: pip install rembg pillow")

    imgs = list_images(IN_DIR)
    if not imgs:
        print(f"[WARN] No hay imágenes en {IN_DIR}")
        return

    ok = 0; skipped = 0
    for i, path in enumerate(imgs, 1):
        img = cv2.imread(path)
        if img is None:
            print(f"[{i}/{len(imgs)}] ERROR al leer {os.path.basename(path)}")
            skipped += 1; continue

        H0, W0 = img.shape[:2]
        if W0 > MAX_WIDTH:
            s = MAX_WIDTH / float(W0)
            work = cv2.resize(img, (int(W0*s), int(H0*s)), interpolation=cv2.INTER_AREA)
        else:
            work = img
        h, w = work.shape[:2]

        base = os.path.splitext(os.path.basename(path))[0]
        per_img = 0

        if BACKEND == "rembg":
            rgba, cov = rembg_cutout(img)
            if rgba is not None and cov >= AREA_RATIO_THRESH:
                out_name = f"{base}_cut01.png"
                cv2.imwrite(os.path.join(out_dir, out_name), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
                per_img = 1
            else:
                print(f"[{i}/{len(imgs)}] Rembg < {AREA_RATIO_THRESH*100:.0f}%: {os.path.basename(path)}")
        else:
            # SAM flow
            generator = build_mask_generator(sam, h, w)
            with torch.no_grad():
                masks = generator.generate(cv2.cvtColor(work, cv2.COLOR_BGR2RGB))

            if len(masks) == 0:
                if USE_REMBG_FALLBACK_IF_SAM_FAILS and REMBG_AVAILABLE:
                    rgba, cov = rembg_cutout(img)
                    if rgba is not None and cov >= AREA_RATIO_THRESH:
                        out_name = f"{base}_cut01.png"
                        cv2.imwrite(os.path.join(out_dir, out_name), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
                        per_img = 1
                    else:
                        skipped += 1
                        print(f"[{i}/{len(imgs)}] sin máscaras SAM ni fallback >= {AREA_RATIO_THRESH*100:.0f}%: {os.path.basename(path)}")
                else:
                    skipped += 1
                    print(f"[{i}/{len(imgs)}] sin máscaras: {os.path.basename(path)}")
            else:
                # Top-N por área -> reescalar a tamaño original, filtrar por 70% y rankear por área+centrado
                masks_sorted = sorted(masks, key=mask_area, reverse=True)[:20]
                bool_masks, scores = [], []
                min_pixels = int(AREA_RATIO_THRESH * W0 * H0)
                for m in masks_sorted:
                    seg_small = m["segmentation"].astype(np.uint8)
                    seg_orig = cv2.resize(seg_small, (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)
                    if int(seg_orig.sum()) < min_pixels:
                        continue
                    area_norm = seg_orig.sum() / float(W0 * H0)
                    center    = score_centered(seg_orig, W0, H0)
                    score     = (1.0 - CENTER_BONUS) * area_norm + CENTER_BONUS * center
                    bool_masks.append(seg_orig); scores.append(score)

                if not bool_masks and USE_REMBG_FALLBACK_IF_SAM_FAILS and REMBG_AVAILABLE:
                    rgba, cov = rembg_cutout(img)
                    if rgba is not None and cov >= AREA_RATIO_THRESH:
                        out_name = f"{base}_cut01.png"
                        cv2.imwrite(os.path.join(out_dir, out_name), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
                        per_img = 1
                    else:
                        skipped += 1
                        print(f"[{i}/{len(imgs)}] todas < {AREA_RATIO_THRESH*100:.0f}% y fallback < umbral: {os.path.basename(path)}")
                elif bool_masks:
                    keep_idx = nms_by_iou(bool_masks, scores, IOU_SUPPRESS)[:MAX_CUTS_PER_IMG]
                    for k, idx in enumerate(keep_idx, 1):
                        rgba = to_rgba_crop(img, bool_masks[idx])
                        if rgba is None: continue
                        out_name = f"{base}_cut{k:02d}.png"
                        cv2.imwrite(os.path.join(out_dir, out_name), rgba)
                        per_img += 1

        ok += 1
        print(f"[{i}/{len(imgs)}] {os.path.basename(path)} -> {per_img} cut-outs (>= {AREA_RATIO_THRESH*100:.0f}%)")

    install_latest_atomic(out_root)
    print(f"\n[RESUMEN] total:{len(imgs)} ok:{ok} saltadas:{skipped}")
    print(f"[OK] Cut-outs: {os.path.join(out_root, OUT_SUB)}")
    print(f"[OK] Últimos:  {os.path.join(OUT_PARENT, 'latest', OUT_SUB)}")

if __name__ == "__main__":
    main()
