# -*- coding: utf-8 -*-
import argparse, os, json, csv, shutil, glob
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_tf(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def load_model(weights_path, num_classes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(device).eval()
    return model, device

def compose_rgb(path):
    img = Image.open(path).convert("RGBA")
    bg = Image.new("RGB", img.size, (255,255,255))
    bg.paste(img, mask=img.split()[-1])
    return bg

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--classmap", required=True, help="class_to_idx.json")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--output_dir", default=None, help="si se define y --move, creará subcarpetas por clase")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--move", action="store_true", help="mover (en vez de copiar) a subcarpetas por clase")
    args = ap.parse_args()

    with open(args.classmap, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v:k for k,v in class_to_idx.items()}

    model, device = load_model(args.weights, num_classes=len(idx_to_class))
    tf = build_tf(args.img_size)

    # buscar imágenes
    exts = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")
    files = [p for p in glob.glob(os.path.join(args.input_dir, "**/*"), recursive=True) if p.lower().endswith(exts)]
    if not files:
        print(f"No se hallaron imágenes en {args.input_dir}")
        return

    ensure_dir(Path(args.out_csv).parent)
    if args.output_dir:
        ensure_dir(args.output_dir)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["file","pred_label","prob_pred"])
        for i, fp in enumerate(sorted(files), 1):
            img = compose_rgb(fp)
            x = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                prob = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                pred_idx = int(prob.argmax())
                pred_lab = idx_to_class[pred_idx]
                conf = float(prob[pred_idx])

            wr.writerow([fp, pred_lab, f"{conf:.4f}"])
            print(f"[{i}/{len(files)}] {os.path.basename(fp)} -> {pred_lab} ({conf:.3f})")

            if args.output_dir:
                dest_dir = Path(args.output_dir)/pred_lab
                ensure_dir(dest_dir)
                dest = dest_dir/Path(fp).name
                if args.move:
                    shutil.move(fp, dest)
                else:
                    shutil.copy2(fp, dest)

if __name__ == "__main__":
    main()
