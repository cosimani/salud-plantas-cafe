# -*- coding: utf-8 -*-
import os, cv2, numpy as np, torch, glob
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ====== CONFIGURACIÓN ======
ROOT_DIR = r"C:/Cosas/2025/IBERO2025/dataset-hojas-recortes - wetransfer 27jul2025/dataset-hojas-recortes"
MODEL_TYPE = "vit_b"  # "vit_b" recomendado (rápido). Podés usar "vit_h" si tenés VRAM.
CHECKPOINT = r"C:/Cosas/2025/IBERO2025/SAM/sam_vit_b_01ec64.pth"

# Tamaño de trabajo para SAM (se reescala la máscara a original luego)
MAX_WIDTH = 1280  # 1024–1536 suele andar bien
# Umbral área mínima relativo al tamaño de la imagen de trabajo (para descartar ruido)
MIN_AREA_FRAC = 0.002  # 0.2% del área; ajustá 0.001–0.005 según ruido

# Extensiones válidas
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_subfolders(path):
    return [d for d in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, d))]

def list_images(path):
    files = []
    for ext in EXTS:
        files.extend(glob.glob(os.path.join(path, f"*{ext}")))
        files.extend(glob.glob(os.path.join(path, f"*{ext.upper()}")))
    return sorted(files)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def largest_mask(masks):
    """Devuelve la máscara (bool) de mayor área, o None si lista vacía."""
    if not masks: 
        return None
    areas = [m["area"] if "area" in m else int(m["segmentation"].sum()) for m in masks]
    return masks[int(np.argmax(areas))]["segmentation"].astype(np.uint8)

def tight_crop_rgba(img_bgr, mask_bin):
    """Crea recorte RGBA (fondo transparente) ajustado al bbox de la máscara."""
    # limpieza ligera para bordes más prolijos
    kernel = np.ones((3,3), np.uint8)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)

    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return None  # sin píxeles

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    crop_bgr = img_bgr[y1:y2+1, x1:x2+1]
    crop_msk = mask_bin[y1:y2+1, x1:x2+1]

    h, w = crop_bgr.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = crop_bgr
    rgba[..., 3] = crop_msk  # alpha

    return rgba

def build_mask_generator(sam, work_img_shape):
    H, W = work_img_shape[:2]
    min_area = int(MIN_AREA_FRAC * (H * W))
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=24,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=min_area
    )

def process_image(img_path, mask_generator):
    """Procesa una imagen: devuelve el RGBA recortado o None si falla."""
    # Leer original (BGR)
    orig = cv2.imread(img_path)
    if orig is None:
        return None

    H0, W0 = orig.shape[:2]

    # Redimensionado para SAM
    if W0 > MAX_WIDTH:
        s = MAX_WIDTH / float(W0)
        work = cv2.resize(orig, (int(W0*s), int(H0*s)), interpolation=cv2.INTER_AREA)
    else:
        work = orig.copy()

    work_rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        masks = mask_generator.generate(work_rgb)

    if len(masks) == 0:
        return None

    # Máscara más grande en tamaño de trabajo
    m_work = largest_mask(masks)
    if m_work is None:
        return None

    # Reescala máscara al tamaño original (nearest para conservar bordes)
    m_orig = cv2.resize(m_work, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)
    m_orig = (m_orig > 0).astype(np.uint8) * 255

    # Crear RGBA recortado al bbox
    rgba = tight_crop_rgba(orig, m_orig)
    return rgba

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Cargando SAM {MODEL_TYPE} en {device}...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT).to(device)

    # Creamos un generador “base” y lo reconstruimos por cada imagen según tamaño (min_area depende del tamaño)
    # Para performance podrías crear por tamaño, pero aquí lo hacemos simple y seguro.
    total_imgs, ok, skipped = 0, 0, 0

    subfolders = list_subfolders(ROOT_DIR)
    if not subfolders:
        print(f"[WARN] No se encontraron subcarpetas en: {ROOT_DIR}")
        return

    for sub in subfolders:
        in_dir = os.path.join(ROOT_DIR, sub)
        out_dir = os.path.join(ROOT_DIR, f"{sub}_segmentado")
        ensure_dir(out_dir)
        print(f"\n[CARPETA] {sub} -> {sub}_segmentado")

        imgs = list_images(in_dir)
        if not imgs:
            print("  (sin imágenes)")
            continue

        for i, img_path in enumerate(imgs, 1):
            total_imgs += 1
            name = os.path.basename(img_path)
            out_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".png")

            # Evitar reprocesar si existe
            if os.path.exists(out_path):
                print(f"  [{i}/{len(imgs)}] skip (existe) {name}")
                ok += 1
                continue

            # Construir mask_generator con el tamaño de trabajo de esta imagen
            # (para que min_area sea relativo a cada imagen)
            tmp = cv2.imread(img_path)
            if tmp is None:
                print(f"  [{i}/{len(imgs)}] ERROR al leer {name}")
                skipped += 1
                continue

            # Redimensionado provisional para parámetros
            H0, W0 = tmp.shape[:2]
            if W0 > MAX_WIDTH:
                s = MAX_WIDTH / float(W0)
                work_shape = (int(H0*s), int(W0*s), 3)
            else:
                work_shape = tmp.shape

            mask_generator = build_mask_generator(sam, work_shape)

            try:
                rgba = process_image(img_path, mask_generator)
                if rgba is None:
                    print(f"  [{i}/{len(imgs)}] sin máscara válida: {name}")
                    skipped += 1
                    continue
                cv2.imwrite(out_path, rgba)
                print(f"  [{i}/{len(imgs)}] OK -> {os.path.basename(out_path)}")
                ok += 1
            except Exception as e:
                print(f"  [{i}/{len(imgs)}] ERROR {name}: {e}")
                skipped += 1

    print(f"\n[RESUMEN] procesadas: {total_imgs}, ok: {ok}, saltadas: {skipped}")

if __name__ == "__main__":
    main()
