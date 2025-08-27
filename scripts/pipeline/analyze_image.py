# -*- coding: utf-8 -*-
"""
Pipeline completo:
- Detecta hojas con YOLOv8
- Recorta cada hoja (opcional: --save_crops)
- (Opcional) Segmenta cada crop con SAM y guarda PNG RGBA (fondo transparente) (--save_sam)
- Clasifica cada crop (healthy vs affected)
- Guarda imagen anotada + CSV por imagen y summary global

Ejemplos:
# Imagen única + crops
python scripts/pipeline/analyze_image.py --image data/yolo/images/test/test_0000.jpg --yolo_weights runs/detect/train/weights/best.pt --clf_weights runs/classify/resnet18/best.pt --out_dir runs/pipeline --yolo_imgsz 960 --yolo_conf 0.40 --save_crops

# Imagen única + crops + SAM + separación por clase
python scripts/pipeline/analyze_image.py --image data/yolo/images/test/test_0000.jpg --yolo_weights runs/detect/train/weights/best.pt --clf_weights runs/classify/resnet18/best.pt --out_dir runs/pipeline --yolo_imgsz 960 --yolo_conf 0.40 --save_crops --save_sam --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth --sam_model vit_b --sam_max_width 256 --sam_min_area_frac 0.20 --separate_by_class

# Carpeta entera
python scripts/pipeline/analyze_image.py --dir data/yolo/images/test --yolo_weights runs/detect/train/weights/best.pt --clf_weights runs/classify/resnet18/best.pt --out_dir runs/pipeline --save_crops --save_sam --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth --sam_model vit_b --sam_max_width 256 --sam_min_area_frac 0.20 --separate_by_class
"""
import os, argparse, uuid, csv, glob, urllib.request
from pathlib import Path
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO

# --------- SAM (opcional) ----------
SAM_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}

def try_import_sam():
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        return sam_model_registry, SamAutomaticMaskGenerator
    except Exception as e:
        raise ImportError("No se pudo importar 'segment_anything'. Instalalo con:\n"
                          "pip install git+https://github.com/facebookresearch/segment-anything.git") from e

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
    model_name = args.get("model", "resnet18")
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, len(classes))
    else:
        model = models.resnet18(weights=None)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    img_size = args.get("img_size", 384)
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return model, classes, tf

def classify_image(model, tf, device, pil_img):
    x = tf(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    score, idx = torch.max(probs, dim=0)
    return int(idx.item()), float(score.item())

# --------- SAM helpers ----------
def download_checkpoint_if_needed(ckpt_path, model_type):
    if Path(ckpt_path).exists():
        return
    url = SAM_URLS.get(model_type)
    if not url:
        raise ValueError(f"No hay URL para SAM '{model_type}'")
    ensure_dir(Path(ckpt_path).parent)
    print(f"[SAM] Descargando checkpoint {model_type} ...")
    urllib.request.urlretrieve(url, ckpt_path)
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
        points_per_side=16,            # adecuado para crops chicos
        pred_iou_thresh=0.90,
        stability_score_thresh=0.95,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=min_area
    )

def segment_crop_with_sam(sam_modules, sam, bgr_crop, max_width=256, min_area_frac=0.20, coverage_full=0.88):
    # sam_modules: (sam_model_registry, SamAutomaticMaskGenerator)
    _, SamAutomaticMaskGenerator = sam_modules
    H0, W0 = bgr_crop.shape[:2]
    if W0 > max_width:
        s = max_width / float(W0)
        work = cv2.resize(bgr_crop, (int(W0*s), int(H0*s)), interpolation=cv2.INTER_AREA)
    else:
        work = bgr_crop

    work_rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    mask_gen = build_mask_generator(SamAutomaticMaskGenerator, sam, work.shape, min_area_frac)

    with torch.no_grad():
        masks = mask_gen.generate(work_rgb)

    if len(masks) == 0:
        return None  # no mask

    m_work = largest_mask(masks)
    if m_work is None:
        return None

    m_orig_bool = cv2.resize(m_work.astype('uint8'), (W0, H0), interpolation=cv2.INTER_NEAREST) > 0
    m_orig_255 = clean_mask(m_orig_bool)

    # Si cubre casi todo, devolvemos el crop completo con alpha 255
    coverage = float((m_orig_255 > 0).sum()) / float(m_orig_255.size)
    if coverage >= coverage_full:
        rgba = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2BGRA)
        rgba[..., 3] = 255
        return rgba

    ys, xs = (m_orig_255 > 0).nonzero()
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    crop_bgr = bgr_crop[y1:y2+1, x1:x2+1]
    crop_msk = m_orig_255[y1:y2+1, x1:x2+1]
    rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = crop_msk
    return rgba

# --------- Proceso por imagen ----------
def process_one_image(image_path, yolo, clf, cls_names, clf_tf, device, args,
                      dirs, wr_summary, sam_ctx=None):
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
        verbose=False
    )
    r = next(iter(results))
    boxes = r.boxes

    annotated = img_bgr.copy()
    total, healthy_cnt, affected_cnt = 0,0,0

    csv_path = Path(args.out_dir) / f"{stem}_analysis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow([
            "orig_file","crop_file","sam_file","det_conf",
            "x1","y1","x2","y2","class","cls_score",
            "img_w","img_h"
        ])

        if boxes is not None and len(boxes)>0:
            for b in boxes:
                xyxy = b.xyxy[0].tolist()
                det_conf = float(b.conf[0]) if b.conf is not None else 0.0
                x1,y1,x2,y2 = xyxy

                crop = crop_with_pad(img_bgr, x1,y1,x2,y2, pad=args.pad)
                if crop is None: 
                    continue

                # Preferimos segmentar antes de clasificar (si está activado), para clasificar sobre PNG RGBA
                sam_path_str = ""
                rgba_for_clf = None
                if args.save_sam and sam_ctx is not None:
                    rgba = segment_crop_with_sam(
                        sam_ctx['modules'], sam_ctx['sam'], crop,
                        max_width=args.sam_max_width,
                        min_area_frac=args.sam_min_area_frac,
                    )
                    if rgba is not None:
                        # Para separar por clase necesitamos pred_name primero -> clasificamos luego y escribimos
                        rgba_for_clf = rgba

                # Clasificar (si tenemos RGBA de SAM lo convertimos a RGB; sino usamos el crop BGR)
                if rgba_for_clf is not None:
                    pil_input = Image.fromarray(cv2.cvtColor(rgba_for_clf[...,:3], cv2.COLOR_BGR2RGB))
                else:
                    pil_input = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

                pred_idx, pred_score = classify_image(clf, clf_tf, device, pil_input)
                pred_name = cls_names[pred_idx]

                total += 1
                if pred_name.lower().startswith("heal"):
                    healthy_cnt += 1; color=(0,200,0)
                else:
                    affected_cnt += 1; color=(0,0,255)

                # Guardar crop de YOLO (opcional, con separación por clase si se pidió)
                crop_path_str = ""
                if args.save_crops:
                    cls_folder = "healthy" if pred_name.lower().startswith("heal") else "affected"
                    out_dir_crops = (dirs['crops'] / cls_folder) if args.separate_by_class else dirs['crops']
                    crop_name = f"{stem}_{uuid.uuid4().hex[:8]}_yolocrop.jpg"
                    crop_path_str = str(out_dir_crops / crop_name)
                    cv2.imwrite(crop_path_str, crop)

                # Guardar SAM (si está activado), ubicando según clase
                if args.save_sam and rgba_for_clf is not None:
                    cls_folder = "healthy" if pred_name.lower().startswith("heal") else "affected"
                    out_dir_sam = (dirs['sam'] / cls_folder) if args.separate_by_class else dirs['sam']
                    ensure_dir(out_dir_sam)
                    sam_name = f"{stem}_{uuid.uuid4().hex[:8]}_sam.png"
                    sam_path_str = str(out_dir_sam / sam_name)
                    cv2.imwrite(sam_path_str, rgba_for_clf)

                # Anotar en imagen
                label = f"{pred_name} {pred_score:.2f} | det {det_conf:.2f}"
                draw_box_label(annotated, (x1,y1,x2,y2), label, color)

                wr.writerow([image_path, crop_path_str, sam_path_str, f"{det_conf:.4f}",
                             int(x1),int(y1),int(x2),int(y2), pred_name, f"{pred_score:.4f}", W,H])

    out_vis = dirs['predict'] / f"{stem}_analysis.jpg"
    cv2.imwrite(str(out_vis), annotated)

    wr_summary.writerow([image_path, total, healthy_cnt, affected_cnt])
    print(f"[{stem}] hojas={total}, healthy={healthy_cnt}, affected={affected_cnt}")
    return total, healthy_cnt, affected_cnt

# --------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="Ruta a una sola imagen")
    ap.add_argument("--dir", help="Ruta a una carpeta de imágenes")
    ap.add_argument("--yolo_weights", required=True, help="Pesos YOLOv8")
    ap.add_argument("--clf_weights", required=True, help="Pesos clasificador")
    ap.add_argument("--out_dir", default="runs/pipeline")
    ap.add_argument("--yolo_imgsz", type=int, default=960)
    ap.add_argument("--yolo_conf", type=float, default=0.25)
    ap.add_argument("--yolo_iou", type=float, default=0.45)
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--save_crops", action="store_true", help="Guarda recortes de YOLO")
    ap.add_argument("--separate_by_class", action="store_true", help="Guarda crops/SAM en subcarpetas healthy/affected")

    # SAM opcional
    ap.add_argument("--save_sam", action="store_true", help="Segmenta cada crop con SAM y guarda PNG RGBA")
    ap.add_argument("--sam_checkpoint", default=None, help="Ruta al checkpoint SAM (.pth). Si no existe, se descarga.")
    ap.add_argument("--sam_model", default="vit_b", choices=["vit_b","vit_l","vit_h"], help="Variante SAM")
    ap.add_argument("--sam_max_width", type=int, default=256, help="Máx. ancho de trabajo SAM (no upsamplea)")
    ap.add_argument("--sam_min_area_frac", type=float, default=0.20, help="Área mínima relativa para filtrar ruido")

    args = ap.parse_args()

    if not args.image and not args.dir:
        raise ValueError("Debe especificar --image o --dir")

    # Checks y salidas
    file_must_exist(args.yolo_weights, "pesos YOLO")
    file_must_exist(args.clf_weights, "pesos clasificador")
    ensure_dir(args.out_dir)
    dirs = {
        'crops':   Path(args.out_dir) / "crops",
        'sam':     Path(args.out_dir) / "sam",
        'predict': Path(args.out_dir) / "predict",
    }
    if args.save_crops: ensure_dir(dirs['crops'])
    if args.save_sam:   ensure_dir(dirs['sam'])
    ensure_dir(dirs['predict'])

    # Si se separa por clase, preparar subcarpetas
    if args.separate_by_class:
        if args.save_crops:
            ensure_dir(dirs['crops'] / "healthy"); ensure_dir(dirs['crops'] / "affected")
        if args.save_sam:
            ensure_dir(dirs['sam'] / "healthy"); ensure_dir(dirs['sam'] / "affected")

    # Modelos
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    yolo = YOLO(args.yolo_weights)
    clf, cls_names, clf_tf = load_classifier(args.clf_weights, device)

    # SAM (opcional)
    sam_ctx = None
    if args.save_sam:
        if not args.sam_checkpoint:
            args.sam_checkpoint = str(Path("checkpoints") / f"sam_{args.sam_model}.pth")
        download_checkpoint_if_needed(args.sam_checkpoint, args.sam_model)
        sam_model_registry, SamAutomaticMaskGenerator = try_import_sam()
        print(f"[SAM] Cargando {args.sam_model} ...")
        sam = sam_model_registry[args.sam_model](checkpoint=args.sam_checkpoint).to(device)
        sam_ctx = {'modules': (sam_model_registry, SamAutomaticMaskGenerator), 'sam': sam}

    # CSV resumen global
    summary_path = Path(args.out_dir) / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as fs:
        wr_sum = csv.writer(fs)
        wr_sum.writerow(["file","total","healthy","affected","pct_healthy","pct_affected"])
        total_all, healthy_all, affected_all = 0,0,0

        # Imagen única
        if args.image:
            file_must_exist(args.image, "imagen de entrada")
            t,h,a = process_one_image(args.image, yolo, clf, cls_names, clf_tf, device, args, dirs, wr_sum, sam_ctx)
            total_all+=t; healthy_all+=h; affected_all+=a

        # Carpeta completa
        if args.dir:
            exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")
            imgs = [p for p in glob.glob(os.path.join(args.dir,"*")) if p.lower().endswith(exts)]
            for p in imgs:
                t,h,a = process_one_image(p, yolo, clf, cls_names, clf_tf, device, args, dirs, wr_sum, sam_ctx)
                total_all+=t; healthy_all+=h; affected_all+=a

        # resumen global y escribir última fila TOTAL con porcentajes
        pct_h = (healthy_all/total_all*100.0) if total_all>0 else 0.0
        pct_a = (affected_all/total_all*100.0) if total_all>0 else 0.0
        wr_sum.writerow(["TOTAL", total_all, healthy_all, affected_all, f"{pct_h:.2f}", f"{pct_a:.2f}"])

    print("\n[RESUMEN GLOBAL]")
    print(f"Total hojas: {total_all}")
    print(f"Healthy: {healthy_all}")
    print(f"Affected: {affected_all}")
    print(f"Resultados guardados en {summary_path}")

if __name__ == "__main__":
    main()
