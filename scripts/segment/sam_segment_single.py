# -*- coding: utf-8 -*-
"""
Segmentación de hojas con SAM (carpeta única de entrada) → PNG RGBA con fondo transparente.
Uso:
python scripts/segment/sam_segment_single.py \
  --input data/crops \
  --output data/sam \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model vit_b \
  --max_width 1280 \
  --min_area_frac 0.002

Requisitos:
- segment-anything (SAM) de Meta: pip install git+https://github.com/facebookresearch/segment-anything.git
- torch/torchvision (con o sin CUDA, según tu máquina)
"""
import argparse, os, glob, numpy as np, cv2, torch, urllib.request
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SAM_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
}

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def list_images(path):
    files = []
    for ext in EXTS:
        files.extend(glob.glob(os.path.join(path, f"*{ext}")))
        files.extend(glob.glob(os.path.join(path, f"*{ext.upper()}")))
    return sorted(files)

def largest_mask(masks):
    if not masks: return None
    areas = [m.get("area", int(m["segmentation"].sum())) for m in masks]
    return masks[int(np.argmax(areas))]["segmentation"].astype(np.uint8)

def tight_crop_rgba(img_bgr, mask_bin):
    kernel = np.ones((3,3), np.uint8)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0: return None
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    crop_bgr = img_bgr[y1:y2+1, x1:x2+1]
    crop_msk = mask_bin[y1:y2+1, x1:x2+1]
    h, w = crop_bgr.shape[:2]
    rgba = np.zeros((h,w,4), dtype=np.uint8)
    rgba[...,:3] = crop_bgr
    rgba[..., 3] = crop_msk
    return rgba

def build_mask_generator(sam, work_shape, min_area_frac):
    H, W = work_shape[:2]
    min_area = int(min_area_frac * (H*W))
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=24,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=min_area
    )


def process_image(img_path, mask_generator, max_width):
    orig = cv2.imread(img_path)
    if orig is None: return None
    H0, W0 = orig.shape[:2]
    if W0 > max_width:
        s = max_width / float(W0)
        work = cv2.resize(orig, (int(W0*s), int(H0*s)), interpolation=cv2.INTER_AREA)
    else:
        work = orig.copy()
    work_rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        masks = mask_generator.generate(work_rgb)
    if len(masks) == 0: return None
    m_work = largest_mask(masks)
    if m_work is None: return None
    m_orig = cv2.resize(m_work, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)
    m_orig = (m_orig > 0).astype(np.uint8) * 255
    return tight_crop_rgba(orig, m_orig)


def download_checkpoint(checkpoint_path, model_type):
    url = SAM_URLS.get(model_type)
    if not url:
        raise ValueError(f"No URL definida para modelo {model_type}")
    ensure_dir(os.path.dirname(checkpoint_path))
    print(f"[INFO] Descargando checkpoint {model_type} desde {url} ...")
    urllib.request.urlretrieve(url, checkpoint_path)
    print(f"[OK] Guardado en {checkpoint_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Carpeta con imágenes (crops de YOLO)")
    ap.add_argument("--output", required=True, help="Carpeta de salida de PNG RGBA")
    ap.add_argument("--checkpoint", required=True, help="ruta al .pth de SAM (vit_b/h/l)")
    ap.add_argument("--model", default="vit_b", choices=["vit_b","vit_l","vit_h"], help="tipo de SAM")
    ap.add_argument("--max_width", type=int, default=1280, help="reescala de trabajo para SAM")
    ap.add_argument("--min_area_frac", type=float, default=0.002, help="umbral de área mínima relativa")
    args = ap.parse_args()

    # Descarga automática si falta el checkpoint
    if not os.path.exists(args.checkpoint):
        download_checkpoint(args.checkpoint, args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Cargando SAM {args.model} en {device} ...")
    sam = sam_model_registry[args.model](checkpoint=args.checkpoint).to(device)

    ensure_dir(args.output)
    imgs = list_images(args.input)
    if not imgs:
        print(f"[WARN] No se encontraron imágenes en {args.input}")
        return

    ok, skipped = 0, 0
    for i, p in enumerate(imgs, 1):
        name = os.path.basename(p)
        out_path = os.path.join(args.output, os.path.splitext(name)[0] + ".png")
        if os.path.exists(out_path):
            print(f"[{i}/{len(imgs)}] skip (existe) {name}")
            ok += 1
            continue

        tmp = cv2.imread(p)
        if tmp is None:
            print(f"[{i}/{len(imgs)}] ERROR al leer {name}")
            skipped += 1
            continue
        H0, W0 = tmp.shape[:2]
        if W0 > args.max_width:
            s = args.max_width / float(W0)
            work_shape = (int(H0*s), int(W0*s), 3)
        else:
            work_shape = tmp.shape

        mask_gen = build_mask_generator(sam, work_shape, args.min_area_frac)
        try:
            rgba = process_image(p, mask_gen, args.max_width)
            if rgba is None:
                print(f"[{i}/{len(imgs)}] sin máscara válida: {name}")
                skipped += 1
                continue
            cv2.imwrite(out_path, rgba)
            print(f"[{i}/{len(imgs)}] OK -> {os.path.basename(out_path)}")
            ok += 1
        except Exception as e:
            print(f"[{i}/{len(imgs)}] ERROR {name}: {e}")
            skipped += 1

    print(f"[RESUMEN] total: {len(imgs)}, ok: {ok}, saltadas: {skipped}")

if __name__ == "__main__":
    main()
