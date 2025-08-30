# -*- coding: utf-8 -*-
"""
Clasifica con ResNet todos los crops YOLO que NO tienen predicción desde SAM,
y APPENDEA al predictions.csv existente. También copia los crops a las carpetas
de clase en runs/classify/predict/latest/<clase>/.

Uso (sin parámetros):
    python scripts/classify/predict_missing_yolocrops.py
"""

import os, csv, glob, shutil, json
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

CROPS_MANIFEST = Path("runs/predict/latest/crops/crops_manifest.csv")
CROPS_DIR      = Path("runs/predict/latest/crops")
CLF_LATEST     = Path("runs/classify/predict/latest")
PRED_CSV       = CLF_LATEST / "predictions.csv"
SUMMARY_CSV    = CLF_LATEST / "summary_counts.csv"

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def rgba_to_rgb(img: Image.Image, bg="black") -> Image.Image:
    if img.mode == "RGBA":
        base = Image.new("RGB", img.size, (0,0,0) if bg=="black" else (255,255,255))
        base.paste(img, mask=img.split()[-1])
        return base
    return img.convert("RGB") if img.mode != "RGB" else img

def build_tf(img_size: int, bg: str):
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Lambda(lambda im: rgba_to_rgb(im, bg=bg)),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def detect_default_weights(explicit: str | None) -> Path:
    if explicit: return Path(explicit)
    p = Path("models/resnet18_best.pt")
    if p.is_file(): return p
    cands = sorted(Path("models").glob("*_best.pt"))
    if cands: return cands[0]
    cands = sorted(Path("runs/classify").glob("*/best.pt"))
    if cands: return cands[0]
    return p

def load_model(weights_path: Path, device: torch.device):
    if not weights_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de pesos: {weights_path}")
    ckpt = torch.load(str(weights_path), map_location=device)
    classes = ckpt.get("classes", ["healthy","affected"])
    args = ckpt.get("args", {})
    model_name = str(args.get("model", "resnet18")).lower()
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
    else:
        model = models.resnet18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, len(classes))
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, classes

def read_known_crop_stems_from_predictions(pred_csv: Path):
    stems = set()
    if not pred_csv.is_file(): return stems
    with open(pred_csv, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            st = Path(row["file"]).stem
            cut_pos = st.rfind("_cut")
            crop_stem = st[:cut_pos] if cut_pos != -1 else st
            stems.add(crop_stem)
    return stems

def list_saved_yolo_crops(manifest: Path):
    todo = []
    if manifest.is_file():
        with open(manifest, newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for row in rd:
                if row.get("status","") != "saved": continue
                p = Path(row["crop_file"])
                if p.is_file(): todo.append(p)
    else:
        exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
        todo = [Path(p) for p in glob.glob(str(CROPS_DIR / "*")) if Path(p).suffix.lower() in exts]
    return sorted(todo)

def append_header_if_missing(csv_path: Path, classes: list, topk: int):
    if csv_path.is_file(): return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        header = (["file","pred_class","pred_score"] +
                  [f"top{i}_class" for i in range(1, topk+1)] +
                  [f"top{i}_score" for i in range(1, topk+1)] +
                  [f"prob_{c}" for c in classes])
        wr.writerow(header)

def recompute_summary(csv_path: Path, classes: list, out_path: Path):
    counts = {c:0 for c in classes}
    if csv_path.is_file():
        with open(csv_path, newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for row in rd:
                c = row.get("pred_class","")
                if c in counts: counts[c] += 1
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["class","count"])
        for c in classes: wr.writerow([c, counts[c]])

def main(weights=None, img_size=384, topk=3, bg="black"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = detect_default_weights(weights)
    model, classes = load_model(weights, device)
    tf = build_tf(img_size, bg)

    ensure_dir(CLF_LATEST)
    for c in classes: ensure_dir(CLF_LATEST / c)

    known = read_known_crop_stems_from_predictions(PRED_CSV)
    yolo_crops = list_saved_yolo_crops(CROPS_MANIFEST)

    missing = [p for p in yolo_crops if p.stem not in known]
    if not missing:
        print("[INFO] No hay crops YOLO faltantes por clasificar.")
        recompute_summary(PRED_CSV, classes, SUMMARY_CSV)
        return

    append_header_if_missing(PRED_CSV, classes, topk)

    with open(PRED_CSV, "a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        for i, p in enumerate(missing, 1):
            try:
                im = Image.open(p)
                x = tf(im).unsqueeze(0).to(device)
                with torch.inference_mode():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)[0]
                k = min(topk, len(classes))
                top = torch.topk(probs, k=k)
                pred_idx = int(top.indices[0].item())
                pred_name = classes[pred_idx]
                pred_score = float(top.values[0].item())

                # copiar crop a carpeta de clase
                shutil.copy2(str(p), str(CLF_LATEST / pred_name / p.name))

                row = [str(p), pred_name, f"{pred_score:.6f}"]
                row += [classes[int(idx.item())] for idx in top.indices]
                row += [f"{float(val.item()):.6f}" for val in top.values]
                row += [f"{float(probs[j].item()):.6f}" for j in range(len(classes))]
                wr.writerow(row)

                print(f"[{i}/{len(missing)}] {p.name} -> {pred_name} ({pred_score:.3f})")
            except Exception as e:
                print(f"[WARN] error con {p.name}: {e}")

    # resumen actualizado
    recompute_summary(PRED_CSV, classes, SUMMARY_CSV)
    print(f"[OK] Añadidas {len(missing)} predicciones faltantes a {PRED_CSV}")

if __name__ == "__main__":
    main()
