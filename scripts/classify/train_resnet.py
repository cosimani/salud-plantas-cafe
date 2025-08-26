# -*- coding: utf-8 -*-
import argparse, os, json, csv, random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class RGBAImageFolder(ImageFolder):
    """ImageFolder que compone PNG RGBA sobre fondo blanco."""
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert("RGBA")
        # Componer sobre blanco
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])  # alpha
        if self.transform is not None:
            bg = self.transform(bg)
        return bg, target

def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="data/sam (healthy/ affected)")
    ap.add_argument("--out_dir", required=True, help="runs/classify_resnet/exp1")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8, help="early stopping")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    train_tf, val_tf = build_transforms(args.img_size)
    full_ds = RGBAImageFolder(args.data_dir, transform=None)  # transform per split

    # Split
    n_total = len(full_ds)
    n_val = int(args.val_split * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_ds.dataset.transform = train_tf
    val_ds.dataset.transform = val_tf

    # Dataloaders
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Modelo
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)  # healthy vs affected
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Guardar mapping de clases
    class_to_idx = full_ds.class_to_idx
    save_json(class_to_idx, os.path.join(args.out_dir, "class_to_idx.json"))
    save_json(vars(args), os.path.join(args.out_dir, "config.json"))

    # Log CSV
    log_path = os.path.join(args.out_dir, "train_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["epoch","train_loss","train_acc","val_loss","val_acc","best"])

    best_acc = -1.0
    best_path = os.path.join(args.out_dir, "best.pt")
    patience_left = args.patience

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_ld, device, criterion, optimizer)
        va_loss, va_acc = evaluate(model, val_ld, device, criterion)

        is_best = va_acc > best_acc
        if is_best:
            best_acc = va_acc
            torch.save({"model": model.state_dict(), "class_to_idx": class_to_idx}, best_path)
            patience_left = args.patience

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            wr = csv.writer(f); wr.writerow([epoch, f"{tr_loss:.4f}", f"{tr_acc:.4f}", f"{va_loss:.4f}", f"{va_acc:.4f}", int(is_best)])

        print(f"[{epoch}/{args.epochs}] train_acc={tr_acc:.3f} val_acc={va_acc:.3f} {'*' if is_best else ''}")

        patience_left -= 1
        if patience_left <= 0:
            print("Early stopping.")
            break

    print(f"Best val_acc: {best_acc:.4f} | Saved: {best_path}")

if __name__ == "__main__":
    main()
