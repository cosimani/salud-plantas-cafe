# -*- coding: utf-8 -*-
import argparse, os, time, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_dataloaders(root, img_size, batch_size, num_workers=2):
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_tf)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_ds, val_ds, train_dl, val_dl

def build_model(name, num_classes, use_pretrained, feature_extraction):
    name = name.lower()
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if use_pretrained else None)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
        if feature_extraction:
            for p in model.parameters(): p.requires_grad = False
            for p in model.fc.parameters(): p.requires_grad = True
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if use_pretrained else None)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
        if feature_extraction:
            for p in model.parameters(): p.requires_grad = False
            for p in model.fc.parameters(): p.requires_grad = True
    else:
        raise ValueError(f"Modelo no soportado: {name}")
    return model

def evaluate(model, dl, device):
    model.eval(); correct=0; total=0; loss_sum=0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x,y in dl:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds==y).sum().item()
            total += x.size(0)
    return loss_sum/total, correct/total

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds, val_ds, train_dl, val_dl = build_dataloaders(args.data_dir, args.img_size, args.batch_size, args.workers)
    class_names = train_ds.classes
    num_classes = len(class_names)
    print("Clases:", class_names)

    model = build_model(args.model, num_classes, args.use_pretrained, args.feature_extraction is True)
    model = model.to(device)

    # Optim y scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = nn.CrossEntropyLoss()

    # Salidas
    run_dir = Path(f"runs/classify/{args.model}")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    meta_path = run_dir / "meta.json"

    best_acc = -1.0
    patience = args.early_stop
    patience_counter = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss=0.0; seen=0
        t0=time.time()
        for x,y in train_dl:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
            seen += x.size(0)
        scheduler.step()

        val_loss, val_acc = evaluate(model, val_dl, device)
        tr_loss = epoch_loss/seen
        dt=time.time()-t0
        print(f"[{epoch:03d}/{args.epochs}] train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | {dt:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "classes": class_names, "args": vars(args)}, best_path)
            patience_counter = 0
            print(f"  -> guardado: {best_path} (val_acc={best_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> early stopping (paciencia={patience})")
                break

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"classes": class_names, "best_acc": best_acc, "args": vars(args)}, f, indent=2, ensure_ascii=False)
    print("Listo. Mejor modelo:", best_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Carpeta con train/ y val/")
    ap.add_argument("--model", default="resnet18", choices=["resnet18","resnet50"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--use_pretrained", action="store_true")
    ap.add_argument("--feature_extraction", action="store_true", help="Si se pasa, congela el backbone y entrena solo la cabeza")
    ap.add_argument("--no_feature_extraction", action="store_true", help="Compatibilidad con README: ignora y hace FT completo")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--early_stop", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    # Compat: si pasan --no_feature_extraction, entonces feature_extraction=False:
    if args.no_feature_extraction:
        args.feature_extraction = False
    train(args)
