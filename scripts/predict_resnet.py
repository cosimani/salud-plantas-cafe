# -*- coding: utf-8 -*-
"""
Clasificación Healthy vs Affected (o las clases del checkpoint) con CSV extendido.
Ejecución sin parámetros:
    python scripts/classify/predict_resnet.py

Por defecto:
- Pesos  : models/resnet18_best.pt  (autodetecta si no está)
- Input  : runs/segment/latest/rgba/
- Output : runs/classify/predict/<timestamp>/{<clases>}/
- Latest : runs/classify/predict/latest/
- CSVs   : predictions.csv (por archivo, con Top-K + prob_<clase>) y summary_counts.csv (resumen)
"""

import argparse, os, csv, shutil, glob, json
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# ---------- FS utils ----------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def safe_copytree(src: str, dst: str) -> None:
    ensure_dir(dst)
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target_root = os.path.join(dst, rel) if rel != "." else dst
        ensure_dir(target_root)
        for d in dirs: ensure_dir(os.path.join(target_root, d))
        for f in files:
            shutil.copy2(os.path.join(root, f), os.path.join(target_root, f))

def install_latest_atomic(run_root: str) -> None:
    parent = os.path.join("runs", "classify", "predict")
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

# ---------- Imagen ----------
def rgba_to_rgb(img: Image.Image, bg="black") -> Image.Image:
    if img.mode == "RGBA":
        bg_rgb = (0,0,0) if str(bg).lower() == "black" else (255,255,255)
        base = Image.new("RGB", img.size, bg_rgb)
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

# ---------- Modelo ----------
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
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, len(classes))
    else:
        model = models.resnet18(weights=None)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, len(classes))

    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, classes, model_name

# ---------- Predicción ----------
def predict_folder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = detect_default_weights(args.weights)
    model, classes, model_name = load_model(weights, device)
    tf = build_tf(args.img_size, args.bg)

    in_dir = Path(args.input)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.output or os.path.join("runs", "classify", "predict", stamp))
    ensure_dir(out_root)

    # subcarpetas por clase
    class_dirs = {c: (out_root / c) for c in classes}
    for d in class_dirs.values(): d.mkdir(parents=True, exist_ok=True)

    # CSVs
    csv_path = out_root / "predictions.csv"
    summary_path = out_root / "summary_counts.csv"
    meta_path = out_root / "meta.json"

    # encabezados: base + topk + prob_<clase>
    prob_headers = [f"prob_{c}" for c in classes]
    header = (["file", "pred_class", "pred_score"] +
              [f"top{i}_class" for i in range(1, args.topk+1)] +
              [f"top{i}_score" for i in range(1, args.topk+1)] +
              prob_headers)

    EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    files = [Path(p) for p in sorted(glob.glob(str(in_dir / "*")))]
    files = [p for p in files if p.suffix.lower() in EXTS]
    if not files:
        print(f"[WARN] No se encontraron imágenes en {in_dir}")
        return

    totals = {c: 0 for c in classes}

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(header)

        for i, p in enumerate(files, 1):
            try:
                im = Image.open(p)
                im = rgba_to_rgb(im, bg=args.bg)
                x = tf(im).unsqueeze(0).to(device)

                with torch.inference_mode():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)[0]  # tensor [C]

                # Top-K
                k = min(args.topk, len(classes))
                topk = torch.topk(probs, k=k)
                pred_idx = int(topk.indices[0].item())
                pred_name = classes[pred_idx]
                pred_score = float(topk.values[0].item())

                # mover/copiar
                dest_path = class_dirs[pred_name] / p.name
                if args.move:
                    shutil.move(str(p), dest_path)
                else:
                    shutil.copy2(p, dest_path)

                # fila CSV
                row = [str(p), pred_name, f"{pred_score:.6f}"]
                row += [classes[int(idx.item())] for idx in topk.indices]
                row += [f"{float(val.item()):.6f}" for val in topk.values]
                # probabilidades por clase (en el orden de 'classes')
                row += [f"{float(probs[j].item()):.6f}" for j in range(len(classes))]
                wr.writerow(row)

                totals[pred_name] += 1
                print(f"[{i}/{len(files)}] {p.name} -> {pred_name} ({pred_score:.3f})")
            except Exception as e:
                print(f"[WARN] error con {p.name}: {e}")

    # resumen por clase
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["class", "count"])
        for c in classes:
            wr.writerow([c, totals.get(c, 0)])

    # meta corrida (útil para el artículo)
    meta = {
        "timestamp": stamp,
        "weights": str(weights),
        "model_name": model_name,
        "classes": classes,
        "img_size": args.img_size,
        "device": str(device),
        "bg": args.bg,
        "topk": args.topk,
        "moved": bool(args.move),
        "input": str(in_dir),
        "output": str(out_root)
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    install_latest_atomic(str(out_root))

    # log final
    print("\nResumen por clase:")
    for c in classes:
        print(f"  {c}: {totals.get(c,0)}")
    print("\nPesos:", weights)
    print("Salida:", out_root)
    print("CSV detalle:", csv_path)
    print("CSV resumen:", summary_path)
    print("Meta:", meta_path)
    print("Latest:", os.path.join("runs", "classify", "predict", "latest"))

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None,
                    help="Ruta a pesos .pt (default: autodetección en models/ o runs/classify/*/best.pt)")
    ap.add_argument("--input", default=os.path.join("runs","segment","latest","rgba"),
                    help="Carpeta con PNGs segmentados (default: runs/segment/latest/rgba)")
    ap.add_argument("--output", default=None,
                    help="Carpeta destino (default: runs/classify/predict/<timestamp>)")
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--move", action="store_true", help="Si se pasa, mueve en lugar de copiar")
    ap.add_argument("--bg", type=str, default="black", choices=["black","white"],
                    help="Color de fondo para componer RGBA→RGB (default: black)")
    args = ap.parse_args()
    predict_folder(args)

# python scripts/predict_resnet.py