# -*- coding: utf-8 -*-
import argparse, os, csv, shutil
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

def load_model(weights_path, device):
    ckpt = torch.load(weights_path, map_location=device)
    classes = ckpt.get("classes", ["healthy","affected"])
    args = ckpt.get("args", {})
    model_name = args.get("model", "resnet18")
    if model_name == "resnet50":
        model = models.resnet50()
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, len(classes))
    else:
        model = models.resnet18()
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, classes

def build_tf(img_size):
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def predict_folder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(args.weights, device)
    tf = build_tf(args.img_size)

    in_dir = Path(args.input); out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "healthy").mkdir(exist_ok=True)
    (out_dir / "affected").mkdir(exist_ok=True)

    csv_path = out_dir / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        header = ["file", "pred", "score"] + [f"top{i}_class" for i in range(1, args.topk+1)] + [f"top{i}_score" for i in range(1, args.topk+1)]
        wr.writerow(header)

        exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
        files = [p for p in in_dir.iterdir() if p.suffix.lower() in exts]
        for i, p in enumerate(sorted(files), 1):
            try:
                im = Image.open(p).convert("RGB")
                x = tf(im).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)[0]
                topk = torch.topk(probs, k=min(args.topk, len(classes)))
                pred_idx = int(topk.indices[0].item())
                pred_name = classes[pred_idx]
                pred_score = float(topk.values[0].item())

                # mover/copiar
                dest_dir = out_dir / pred_name
                dest_path = dest_dir / p.name
                if args.move:
                    shutil.move(str(p), dest_path)
                else:
                    shutil.copy2(p, dest_path)

                # CSV
                row = [str(p), pred_name, f"{pred_score:.4f}"]
                row += [classes[int(idx.item())] for idx in topk.indices]
                row += [f"{float(val.item()):.4f}" for val in topk.values]
                wr.writerow(row)

                print(f"[{i}/{len(files)}] {p.name} -> {pred_name} ({pred_score:.2f})")
            except Exception as e:
                print(f"[WARN] error con {p.name}: {e}")

    print("Listo. Resultados en:", out_dir)
    print("CSV:", csv_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="runs/classify/resnet18/best.pt")
    ap.add_argument("--input", required=True, help="Carpeta con PNG segmentados (p.ej. data/sam)")
    ap.add_argument("--output", required=True, help="Carpeta destino con subcarpetas por clase")
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--move", action="store_true", help="Si se pasa, mueve en lugar de copiar")
    args = ap.parse_args()
    predict_folder(args)
