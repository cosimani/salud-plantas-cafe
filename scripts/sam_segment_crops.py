# scripts/sam_segment_crops.py
# -*- coding: utf-8 -*-
"""
Segmentación de hojas con SAM (crops pequeños) con presets y Top-K cut-outs.
Ahora genera un CSV de métricas por corrida:
  runs/segment/<preset>_<sam_model>_<timestamp>/metrics_cutouts.csv
y lo copia a runs/segment/latest/ junto con los PNG.

Ejemplo:
  python scripts/sam_segment_crops.py --preset ultra_recall_crops --sam_model vit_h \
    --top_k 3 --min_coverage 0.20 --min_solidity 0.88
"""

import os
import cv2
import glob
import csv
import numpy as np
import torch
import urllib.request
import shutil
import argparse
from datetime import datetime
from typing import List
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# --- Rutas / formatos ---
DEFAULT_INPUT = os.path.join("runs", "predict", "latest", "crops")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# --- Filtro de tamaño mínimo (px) ---
# Acepta si:
#   - Rectangular: un lado >= MIN_LONG y el otro >= MIN_SHORT (en cualquier orden), o
#   - Cuadrado/Grande: ambos lados >= MIN_SQ.
MIN_SHORT = 40   # lado corto para formato rectangular
MIN_LONG  = 80   # lado largo para formato rectangular
MIN_SQ    = 50   # mínimo para formato ~cuadrado

def is_valid_crop_size(w: int, h: int) -> bool:
    a, b = sorted((w, h))  # a = corto, b = largo
    return (a >= MIN_SHORT and b >= MIN_LONG) or (a >= MIN_SQ and b >= MIN_SQ)

# --- Modelos SAM soportados y checkpoints oficiales ---
MODEL_CHOICES = ("vit_b", "vit_l", "vit_h")
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
def ckpt_path_for_model(model: str) -> str:
    return os.path.join("models", SAM_CKPT_NAME[model])

# --- PRESETS (abreviado) ---
PRESETS = {
    "high_recall_crops": {
        "generator": dict(
            points_per_side=24, points_per_batch=64,
            pred_iou_thresh=0.86, stability_score_thresh=0.90,
            stability_score_offset=1.0, box_nms_thresh=0.65,
            crop_n_layers=0, crop_nms_thresh=0.7,
            crop_overlap_ratio=512/1500, crop_n_points_downscale_factor=1,
            min_area_frac=0.12, output_mode="binary_mask",
        ),
        "runtime": dict(max_width=320, coverage_full=0.85),
    },
    "tiling_detail": {
        "generator": dict(
            points_per_side=16, points_per_batch=64,
            pred_iou_thresh=0.88, stability_score_thresh=0.92,
            stability_score_offset=1.0, box_nms_thresh=0.7,
            crop_n_layers=1, crop_nms_thresh=0.7,
            crop_overlap_ratio=0.35, crop_n_points_downscale_factor=2,
            min_area_frac=0.10, output_mode="binary_mask",
        ),
        "runtime": dict(max_width=512, coverage_full=0.88),
    },
    "ultra_recall_crops": {
        "generator": dict(
            points_per_side=28, points_per_batch=64,
            pred_iou_thresh=0.86, stability_score_thresh=0.90,
            stability_score_offset=1.0, box_nms_thresh=0.65,
            crop_n_layers=1, crop_nms_thresh=0.7,
            crop_overlap_ratio=0.40, crop_n_points_downscale_factor=2,
            min_area_frac=0.10, output_mode="binary_mask",
        ),
        "runtime": dict(max_width=512, coverage_full=0.85),
    },
    "ultra_tiling_x2": {
        "generator": dict(
            points_per_side=24, points_per_batch=64,
            pred_iou_thresh=0.86, stability_score_thresh=0.90,
            stability_score_offset=1.0, box_nms_thresh=0.65,
            crop_n_layers=2, crop_nms_thresh=0.7,
            crop_overlap_ratio=0.45, crop_n_points_downscale_factor=2,
            min_area_frac=0.08, output_mode="binary_mask",
        ),
        "runtime": dict(max_width=512, coverage_full=0.85),
    },
}
DEFAULT_PRESET = "ultra_recall_crops"

# --- Utils FS ---
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def list_images(path):
    files = []
    for ext in EXTS:
        files.extend(glob.glob(os.path.join(path, f"*{ext}")))
        files.extend(glob.glob(os.path.join(path, f"*{ext.upper()}")))
    return sorted(files)
def download_checkpoint_if_needed(ckpt_path, model_type):
    if os.path.exists(ckpt_path): return
    url = SAM_URLS.get(model_type)
    if not url: raise ValueError(f"No hay URL para modelo {model_type}")
    ensure_dir(os.path.dirname(ckpt_path))
    print(f"[INFO] Descargando SAM {model_type} desde {url} ...")
    urllib.request.urlretrieve(url, ckpt_path)
    print(f"[OK] Guardado en {ckpt_path}")

# --- Métricas y helpers de máscaras ---
def mask_area_norm(m_bool: np.ndarray) -> float:
    return float(m_bool.sum()) / float(m_bool.size)

def centroid_score(m_bool: np.ndarray, W:int, H:int) -> float:
    ys, xs = np.where(m_bool)
    if len(xs) == 0: return 0.0
    cx, cy = xs.mean(), ys.mean()
    dx = abs((cx - W/2) / (W/2))
    dy = abs((cy - H/2) / (H/2))
    dist = np.hypot(dx, dy) / np.hypot(1, 1)   # 0..1
    return 1.0 - float(dist)                    # 1 = centro perfecto

def iou_masks(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 0.0 if union == 0 else float(inter)/float(union)

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

def mask_solidity(m_bool: np.ndarray) -> float:
    m = (m_bool > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    area = sum(float(cv2.contourArea(c)) for c in cnts)
    pts = np.vstack(cnts)
    hull = cv2.convexHull(pts)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 0.0:
        return 0.0
    return area / hull_area

def _run_len_from_corner(m_bool: np.ndarray, corner: str):
    """
    Devuelve (run_x, run_y) = cantidad de píxeles contiguos 'True'
    desde la esquina indicada.
    corner ∈ {'tl','tr','bl','br'}
    """
    H, W = m_bool.shape[:2]
    if corner == 'tl':
        run_x = next((x for x in range(W) if not m_bool[0, x]), W)
        run_y = next((y for y in range(H) if not m_bool[y, 0]), H)
        corner_on = m_bool[0,0]
    elif corner == 'tr':
        run_x = next((W-1-x for x in range(W) if not m_bool[0, W-1-x]), W)
        run_y = next((y for y in range(H) if not m_bool[y, W-1]), H)
        corner_on = m_bool[0, W-1]
    elif corner == 'bl':
        run_x = next((x for x in range(W) if not m_bool[H-1, x]), W)
        run_y = next((H-1-y for y in range(H) if not m_bool[H-1-y, 0]), H)
        corner_on = m_bool[H-1, 0]
    elif corner == 'br':
        run_x = next((W-1-x for x in range(W) if not m_bool[H-1, W-1-x]), W)
        run_y = next((H-1-y for y in range(H) if not m_bool[H-1-y, W-1]), H)
        corner_on = m_bool[H-1, W-1]
    else:
        return 0, 0
    if not corner_on:
        return 0, 0
    return int(run_x), int(run_y)

def is_corner_wedge(mask_bool: np.ndarray, frac_thresh: float = 0.50) -> bool:
    """
    True si la máscara forma una “cuña de esquina”.
    """
    H, W = mask_bool.shape[:2]
    if H == 0 or W == 0:
        return False
    for corner in ('tl','tr','bl','br'):
        run_x, run_y = _run_len_from_corner(mask_bool, corner)
        if run_x >= frac_thresh * W and run_y >= frac_thresh * H:
            return True
    return False

def bbox_from_mask(mask_bool: np.ndarray):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0: return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def to_rgba_crop(img_bgr: np.ndarray, mask_bool: np.ndarray):
    bb = bbox_from_mask(mask_bool)
    if bb is None: return None, None
    x1, y1, x2, y2 = bb
    crop_bgr = img_bgr[y1:y2+1, x1:x2+1]
    crop_msk = (mask_bool[y1:y2+1, x1:x2+1] * 255).astype(np.uint8)
    rgba = np.zeros((crop_bgr.shape[0], crop_bgr.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = crop_bgr
    rgba[..., 3]  = crop_msk
    return rgba, bb

# --- Construcción del generador SAM desde preset ---
def build_mask_generator_from_preset(sam, work_shape, preset_name):
    H, W = work_shape[:2]
    cfg = PRESETS[preset_name]["generator"].copy()
    min_area_frac = cfg.pop("min_area_frac", None)
    min_mask_region_area = int(float(min_area_frac) * (H * W)) if min_area_frac is not None else 0
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=cfg.get("points_per_side", 16),
        points_per_batch=cfg.get("points_per_batch", 64),
        pred_iou_thresh=cfg.get("pred_iou_thresh", 0.88),
        stability_score_thresh=cfg.get("stability_score_thresh", 0.92),
        stability_score_offset=cfg.get("stability_score_offset", 1.0),
        box_nms_thresh=cfg.get("box_nms_thresh", 0.7),
        crop_n_layers=cfg.get("crop_n_layers", 0),
        crop_nms_thresh=cfg.get("crop_nms_thresh", 0.7),
        crop_overlap_ratio=cfg.get("crop_overlap_ratio", 512/1500),
        crop_n_points_downscale_factor=cfg.get("crop_n_points_downscale_factor", 1),
        min_mask_region_area=min_mask_region_area,
        output_mode=cfg.get("output_mode", "binary_mask"),
    )

# --- Procesar una imagen → varios cut-outs (Top-K) ---
def process_image_topk(orig_bgr: np.ndarray,
                       work_bgr: np.ndarray,
                       mask_generator: SamAutomaticMaskGenerator,
                       coverage_full: float,
                       top_k: int,
                       min_coverage: float,
                       min_solidity: float = 0.90,
                       center_bonus: float = 0.20,
                       preselect_top: int = 24,
                       iou_suppress: float = 0.55,
                       reject_corner_wedge: bool = False,
                       corner_frac: float = 0.50):
    """
    Devuelve lista de dicts:
      {'rgba','bbox','coverage','solidity','center_score','score'}
    """
    H0, W0 = orig_bgr.shape[:2]
    work_rgb = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        masks = mask_generator.generate(work_rgb)
    if len(masks) == 0:
        return []

    masks_sorted = sorted(
        masks, key=lambda m: int(m.get("area", int(m["segmentation"].sum()))), reverse=True
    )[:preselect_top]

    bool_masks, scores, covs, sols, centers = [], [], [], [], []
    for m in masks_sorted:
        seg_small = m["segmentation"].astype(np.uint8)
        seg_orig  = cv2.resize(seg_small, (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)

        cov = mask_area_norm(seg_orig)
        if cov < min_coverage:
            continue

        sol = mask_solidity(seg_orig)
        if sol < min_solidity:
            continue

        if reject_corner_wedge and is_corner_wedge(seg_orig, frac_thresh=corner_frac):
            continue  # descarta fondo pegado a esquina

        center = centroid_score(seg_orig, W0, H0)
        score  = (1.0 - center_bonus) * cov + center_bonus * center

        bool_masks.append(seg_orig)
        covs.append(cov); sols.append(sol); centers.append(center); scores.append(score)

    if not bool_masks:
        return []

    best_idx = int(np.argmax(scores))
    best_cov = covs[best_idx]
    if best_cov >= max(coverage_full, 0.95):
        rgba, bb = to_rgba_crop(orig_bgr, bool_masks[best_idx])
        if rgba is None: return []
        return [{
            "rgba": rgba, "bbox": bb, "coverage": covs[best_idx],
            "solidity": sols[best_idx], "center_score": centers[best_idx],
            "score": scores[best_idx]
        }]

    keep = nms_by_iou(bool_masks, scores, iou_suppress)[:top_k]

    out_list = []
    for idx in keep:
        rgba, bb = to_rgba_crop(orig_bgr, bool_masks[idx])
        if rgba is not None:
            out_list.append({
                "rgba": rgba, "bbox": bb, "coverage": covs[idx],
                "solidity": sols[idx], "center_score": centers[idx],
                "score": scores[idx]
            })
    return out_list

# --- Copia atómica "latest" ---
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
    segment_root = os.path.join("runs", "segment")
    ensure_dir("runs"); ensure_dir(segment_root)
    tmp = os.path.join(segment_root, ".latest_tmp")
    latest = os.path.join(segment_root, "latest")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    elif os.path.islink(tmp) or os.path.isfile(tmp):
        try: os.unlink(tmp)
        except Exception: pass
    ensure_dir(tmp)

    # copiar rgba/
    src_sub = os.path.join(run_root, "rgba")
    if os.path.isdir(src_sub):
        dst_sub = os.path.join(tmp, "rgba")
        safe_copytree(src_sub, dst_sub)

    # copiar CSVs de métricas si existen
    for fname in ("metrics_cutouts.csv",):
        src_f = os.path.join(run_root, fname)
        if os.path.isfile(src_f):
            shutil.copy2(src_f, os.path.join(tmp, fname))

    if os.path.islink(latest) or os.path.isfile(latest):
        try: os.unlink(latest)
        except Exception: pass
    elif os.path.isdir(latest):
        shutil.rmtree(latest, ignore_errors=True)
    os.replace(tmp, latest)

def parse_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--preset", type=str, default=None,
                    help=f"Preset SAM a usar. Opciones: {', '.join(PRESETS.keys())}")
    ap.add_argument("--sam_model", type=str, choices=MODEL_CHOICES, default=None,
                    help="Backbone de SAM: vit_b (def), vit_l, vit_h. También via env SAM_MODEL.")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Ruta a checkpoint .pth (opcional). Si no, usa el oficial según --sam_model.")
    ap.add_argument("--top_k", type=int, default=int(os.environ.get("SAM_TOPK", 3)),
                    help="Cantidad de cut-outs por imagen (default 3).")
    ap.add_argument("--min_coverage", type=float, default=float(os.environ.get("SAM_MINCOV", 0.25)),
                    help="Cobertura mínima relativa para aceptar una máscara (default 0.25).")
    ap.add_argument("--min_solidity", type=float, default=float(os.environ.get("SAM_MINSOL", 0.90)),
                    help="Mínimo de convexidad (solidity) para aceptar una máscara (default 0.90).")
    ap.add_argument("--coverage_full", type=float, default=None,
                    help="Override del coverage_full del preset (0..1). Si no se pasa, usa el del preset (con piso 0.95).")
    ap.add_argument("--reject_corner_wedge", action="store_true",
                    help="Si se activa, descarta máscaras con cuña de esquina (fondo).")
    ap.add_argument("--corner_frac", type=float, default=float(os.environ.get("SAM_CORNER_FRAC", 0.50)),
                    help="Fracción mínima (0..1) de ancho/alto para considerar cuña de esquina (default 0.50).")
    ap.add_argument("-h", "--help", action="help", help="Mostrar ayuda y salir")
    return ap.parse_args()

# --- Main ---
def main():
    args = parse_args()
    preset = args.preset or os.environ.get("SAM_PRESET", DEFAULT_PRESET)
    if preset not in PRESETS:
        print(f"[WARN] preset '{preset}' no válido. Usando '{DEFAULT_PRESET}'.")
        preset = DEFAULT_PRESET

    sam_model = (args.sam_model or os.environ.get("SAM_MODEL", "vit_b")).lower()
    if sam_model not in MODEL_CHOICES:
        print(f"[WARN] sam_model '{sam_model}' no válido. Usando 'vit_b'.")
        sam_model = "vit_b"

    gen_cfg = PRESETS[preset]["generator"]
    run_cfg = PRESETS[preset]["runtime"].copy()
    if args.coverage_full is not None:
        run_cfg["coverage_full"] = args.coverage_full

    ensure_dir("runs"); ensure_dir(os.path.join("runs", "segment"))
    run_id   = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = os.path.join("runs", "segment", f"{preset}_{sam_model}_{run_id}")
    rgba_dir = os.path.join(out_root, "rgba")
    ensure_dir(out_root); ensure_dir(rgba_dir)

    # CSV de métricas
    csv_path = os.path.join(out_root, "metrics_cutouts.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow([
            "run_id","preset","sam_model","top_k","min_coverage","min_solidity","coverage_full",
            "reject_corner_wedge","corner_frac",
            "input_file","img_w","img_h","rank","output_png","x1","y1","x2","y2",
            "coverage","solidity","center_score","score",
            "status","reason"
        ])

        # Checkpoint y device
        ckpt_path = args.ckpt or ckpt_path_for_model(sam_model)
        download_checkpoint_if_needed(ckpt_path, sam_model)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[INFO] SAM: {sam_model} | Device: {device}")
        print(f"[INFO] Preset: {preset} | Checkpoint: {ckpt_path}")
        print(f"[INFO] AMG: pps={gen_cfg.get('points_per_side')} "
              f"pred_iou={gen_cfg.get('pred_iou_thresh')} stab={gen_cfg.get('stability_score_thresh')} "
              f"crop_layers={gen_cfg.get('crop_n_layers',0)} min_area_frac={gen_cfg.get('min_area_frac',0)}")
        print(f"[INFO] Runtime: max_width={run_cfg.get('max_width')} coverage_full={run_cfg.get('coverage_full')}")
        print(f"[INFO] Top-K={args.top_k} | min_coverage={args.min_coverage} | "
              f"min_solidity={args.min_solidity} | reject_corner_wedge={args.reject_corner_wedge} "
              f"(corner_frac={args.corner_frac})")

        sam = sam_model_registry[sam_model](checkpoint=ckpt_path).to(device)

        imgs = list_images(DEFAULT_INPUT)
        if not imgs:
            print(f"[WARN] No se encontraron imágenes en {DEFAULT_INPUT}")
            return

        mask_generator = None
        last_shape = None
        ok_imgs, skipped_imgs = 0, 0
        saved_pngs, discarded_pngs = 0, 0

        for i, path in enumerate(imgs, 1):
            name = os.path.basename(path)
            stem = os.path.splitext(name)[0]

            orig = cv2.imread(path)
            if orig is None:
                print(f"[{i}/{len(imgs)}] ERROR al leer {name}")
                skipped_imgs += 1
                continue

            H0, W0 = orig.shape[:2]
            if W0 > run_cfg["max_width"]:
                s = run_cfg["max_width"] / float(W0)
                work = cv2.resize(orig, (int(W0*s), int(H0*s)), interpolation=cv2.INTER_AREA)
            else:
                work = orig

            work_shape = work.shape
            if last_shape != work_shape:
                mask_generator = build_mask_generator_from_preset(sam, work_shape, preset)
                last_shape = work_shape

            try:
                results = process_image_topk(
                    orig_bgr=orig,
                    work_bgr=work,
                    mask_generator=mask_generator,
                    coverage_full=run_cfg["coverage_full"],
                    top_k=args.top_k,
                    min_coverage=args.min_coverage,
                    min_solidity=args.min_solidity,
                    center_bonus=0.20,
                    preselect_top=24,
                    iou_suppress=0.55,
                    reject_corner_wedge=args.reject_corner_wedge,
                    corner_frac=args.corner_frac
                )
                if not results:
                    print(f"[{i}/{len(imgs)}] sin cut-outs válidos: {name}")
                    skipped_imgs += 1
                    continue

                for k, info in enumerate(results, 1):
                    x1, y1, x2, y2 = info["bbox"]
                    crop_w, crop_h = x2 - x1, y2 - y1

                    if not is_valid_crop_size(crop_w, crop_h):
                        # Registrar descartado (no se guarda PNG)
                        wr.writerow([
                            run_id, preset, sam_model, args.top_k, args.min_coverage, args.min_solidity, run_cfg["coverage_full"],
                            int(args.reject_corner_wedge), args.corner_frac,
                            path, W0, H0, k, "", x1, y1, x2, y2,
                            round(info["coverage"], 6), round(info["solidity"], 6),
                            round(info["center_score"], 6), round(info["score"], 6),
                            "discarded", "too_small"
                        ])
                        discarded_pngs += 1
                        print(f"[{i}/{len(imgs)}] {name} cut{k:02d} descartado por tamaño ({crop_w}x{crop_h})")
                        continue

                    out_name = f"{stem}_cut{k:02d}.png"
                    out_path = os.path.join(rgba_dir, out_name)
                    cv2.imwrite(out_path, info["rgba"])

                    wr.writerow([
                        run_id, preset, sam_model, args.top_k, args.min_coverage, args.min_solidity, run_cfg["coverage_full"],
                        int(args.reject_corner_wedge), args.corner_frac,
                        path, W0, H0, k, out_name, x1, y1, x2, y2,
                        round(info["coverage"], 6), round(info["solidity"], 6),
                        round(info["center_score"], 6), round(info["score"], 6),
                        "saved", ""
                    ])
                    saved_pngs += 1

                ok_imgs += 1
                print(f"[{i}/{len(imgs)}] {name} -> {len(results)} cut-outs (saved:{saved_pngs} / discarded:{discarded_pngs})")
            except Exception as e:
                print(f"[{i}/{len(imgs)}] ERROR {name}: {e}")
                skipped_imgs += 1

    install_latest_atomic(out_root)
    print(f"\n[RESUMEN] imgs_total:{len(imgs)} imgs_ok:{ok_imgs} imgs_sin_cuts:{skipped_imgs}")
    print(f"[RESUMEN] pngs_saved:{saved_pngs} pngs_discarded:{discarded_pngs}")
    print(f"[OK] Cut-outs en: {rgba_dir}")
    print(f"[OK] CSV: {csv_path}")
    print(f"[OK] Últimos: runs/segment/latest/ (incluye rgba/ y metrics_cutouts.csv)")

if __name__ == "__main__":
    main()

# Ejemplos:
# python scripts/sam_segment_crops.py --preset ultra_recall_crops --sam_model vit_h --top_k 1 --min_coverage 0.25 --min_solidity 0.88
# python scripts/sam_segment_crops.py --preset ultra_recall_crops --sam_model vit_h --top_k 1 --min_coverage 0.25 --min_solidity 0.88 --reject_corner_wedge --corner_frac 0.50
# python scripts/sam_segment_crops.py --preset ultra_recall_crops --sam_model vit_h --top_k 1 --min_coverage 0.30 --min_solidity 0.88 --reject_corner_wedge --corner_frac 0.60
