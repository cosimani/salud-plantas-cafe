# -*- coding: utf-8 -*-
import argparse, os, time, json, random, csv
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from PIL import Image

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def rgba_to_rgb_pil(img: Image.Image, bg=(0,0,0)) -> Image.Image:
    """Convierte RGBA→RGB usando alpha como máscara (fondo bg)."""
    if img.mode == "RGBA":
        bg_img = Image.new("RGB", img.size, bg)
        bg_img.paste(img, mask=img.split()[-1])
        return bg_img
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

class RGBAtoRGB:
    def __init__(self, bg=(0,0,0)): self.bg = bg
    def __call__(self, img: Image.Image) -> Image.Image:
        return rgba_to_rgb_pil(img, self.bg)

def build_dataloaders(root, img_size, batch_size, num_workers=0, use_rrc=False, bg_rgb=(0,0,0)):
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    train_geo = (transforms.RandomResizedCrop(img_size, scale=(0.8,1.0), ratio=(0.8,1.25))
                 if use_rrc else transforms.Resize((img_size, img_size)))
    train_tf = transforms.Compose([
        RGBAtoRGB(bg_rgb),
        train_geo,
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        RGBAtoRGB(bg_rgb),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_dir = os.path.join(root, "train")
    val_dir   = os.path.join(root, "val")
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_tf)
    return train_ds, val_ds

def make_loaders(train_ds, val_ds, batch_size, num_workers=0, sampler=None, pin_memory=False):
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=num_workers, pin_memory=pin_memory
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    return train_dl, val_dl

def build_model(name, num_classes, use_pretrained, feature_extraction):
    name = name.lower()
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if use_pretrained else None)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if use_pretrained else None)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
    else:
        raise ValueError(f"Modelo no soportado: {name}")
    if feature_extraction:
        for p in model.parameters(): p.requires_grad = False
        for p in model.fc.parameters(): p.requires_grad = True
    return model

@torch.no_grad()
def evaluate(model, dl, device, return_preds=False):
    model.eval(); correct=0; total=0; loss_sum=0.0
    criterion = nn.CrossEntropyLoss()
    all_t, all_p = [], []
    autocast_ok = (device.type == "cuda")
    with torch.amp.autocast('cuda', enabled=autocast_ok):
        for x,y in dl:
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds==y).sum().item()
            total += x.size(0)
            if return_preds:
                all_t.append(y.detach().cpu().numpy())
                all_p.append(preds.detach().cpu().numpy())
    val_loss = loss_sum/total
    val_acc  = correct/total
    if return_preds:
        y_true = np.concatenate(all_t) if all_t else np.zeros((0,), dtype=int)
        y_pred = np.concatenate(all_p) if all_p else np.zeros((0,), dtype=int)
        return val_loss, val_acc, y_true, y_pred
    return val_loss, val_acc

def compute_class_weights(train_ds):
    counts = np.bincount(train_ds.targets)
    weights = 1.0 / np.maximum(counts, 1)
    weights = weights * (len(counts) / weights.sum())
    return torch.tensor(weights, dtype=torch.float32), counts.tolist()

def confusion_from_preds(y_true, y_pred, num_classes):
    M = np.zeros((num_classes, num_classes), dtype=int)
    for t,p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            M[t,p] += 1
    return M

def get_group_lrs(optimizer):
    lrs = [pg.get("lr", 0.0) for pg in optimizer.param_groups]
    # asumimos [backbone, head] si hay dos grupos; si hay uno, duplicamos
    if len(lrs) == 1: lrs = [lrs[0], lrs[0]]
    return lrs[0], lrs[1]

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Respeta lo que venga por --workers. Si el usuario quiere auto, que pase --workers -1
    if args.workers == -1:
        try:
            args.workers = max(os.cpu_count() - 1, 2)
        except Exception:
            args.workers = 2
    # Si pasa 0, queda 0 (sin subprocess en Windows)


    print("Device:", device)

    train_ds, val_ds = build_dataloaders(
        args.data_dir, args.img_size, args.batch_size, args.workers,
        use_rrc=args.use_rrc, bg_rgb=(0,0,0)
    )
    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"Clases ({num_classes}):", class_names)
    print(f"Train imgs: {len(train_ds)} | Val imgs: {len(val_ds)}")

    sampler = None
    class_weights_tensor = None
    if args.auto_class_weights:
        class_weights_tensor, counts = compute_class_weights(train_ds)
        print("Class counts:", counts)
        print("Class weights:", [round(float(w),3) for w in class_weights_tensor])
        class_weights_tensor = class_weights_tensor.to(device)
        # (alternativa sampler comentada)

    train_dl, val_dl = make_loaders(
        train_ds, val_ds, args.batch_size, args.workers, sampler=sampler, pin_memory=args.pin_memory
    )

    model = build_model(args.model, num_classes, args.use_pretrained, args.feature_extraction is True).to(device)

    head_params = list(model.fc.parameters())
    backbone_params = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith("fc.")]
    param_groups = []
    if backbone_params: param_groups.append({"params": backbone_params, "lr": args.lr * args.backbone_lr_scale})
    if head_params:     param_groups.append({"params": head_params, "lr": args.lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Salidas
    run_dir = Path(f"runs/classify/{args.model}")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_run_path = run_dir / "best.pt"
    models_dir = Path("models"); models_dir.mkdir(exist_ok=True, parents=True)
    best_models_path = models_dir / f"{args.model}_best.pt"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ts_copy_path = models_dir / f"{args.model}_{stamp}.pt"
    meta_path = run_dir / "meta.json"
    csv_path  = run_dir / "metrics.csv"

    # CSV: header
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["epoch","train_loss","val_loss","val_acc","lr_backbone","lr_head","dt_sec","best_acc","improved"])

    best_acc = -1.0
    patience = args.early_stop
    patience_counter = 0
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss=0.0; seen=0
        t0=time.time()

        for x,y in train_dl:
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * x.size(0)
            seen += x.size(0)

        # LRs usados en esta época (antes de step del scheduler)
        lr_backbone, lr_head = get_group_lrs(optimizer)

        val_loss, val_acc = evaluate(model, val_dl, device)
        tr_loss = epoch_loss/seen
        dt=time.time()-t0
        print(f"[{epoch:03d}/{args.epochs}] train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | lr_b={lr_backbone:.2e} lr_h={lr_head:.2e} | {dt:.1f}s")

        improved = 0
        if val_acc > best_acc:
            best_acc = val_acc
            payload = {
                "state_dict": model.state_dict(),
                "classes": class_names,
                "args": vars(args),
                "val_acc": float(best_acc),
            }
            torch.save(payload, best_run_path)
            torch.save(payload, best_models_path)
            torch.save(payload, ts_copy_path)
            improved = 1
            patience_counter = 0
            print(f"  -> guardado: {best_run_path} y {best_models_path} (val_acc={best_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> early stopping (paciencia={patience})")

        # Escribimos fila CSV
        with open(csv_path, "a", newline="", encoding="utf-8") as fcsv:
            wr = csv.writer(fcsv)
            wr.writerow([epoch, round(tr_loss,6), round(val_loss,6), round(val_acc,6),
                         lr_backbone, lr_head, round(dt,3), round(best_acc,6), improved])

        # Step scheduler al final de la época
        scheduler.step()

        if patience_counter >= patience:
            break

    # Guardar meta
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "classes": class_names, "best_acc": best_acc, "args": vars(args),
            "best_in_runs": str(best_run_path), "best_in_models": str(best_models_path),
            "timestamp_copy": str(ts_copy_path), "metrics_csv": str(csv_path)
        }, f, indent=2, ensure_ascii=False)

    # (Opcional) matriz de confusión
    if args.save_confusion:
        print("Generando matriz de confusión del conjunto de validación…")
        _, _, y_true, y_pred = evaluate(model, val_dl, device, return_preds=True)
        cm = confusion_from_preds(y_true, y_pred, num_classes)
        cm_csv = run_dir / "confusion_matrix.csv"
        # CSV con encabezados de clases
        with open(cm_csv, "w", newline="", encoding="utf-8") as fcsv:
            wr = csv.writer(fcsv)
            wr.writerow(["true\\pred"] + class_names)
            for i, row in enumerate(cm):
                wr.writerow([class_names[i]] + row.tolist())
        # PNG opcional si tenés matplotlib
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            fig = plt.figure(figsize=(6,5), dpi=150)
            plt.imshow(cm, interpolation="nearest")
            plt.title("Confusion Matrix")
            plt.xlabel("Pred"); plt.ylabel("True")
            plt.xticks(ticks=np.arange(num_classes), labels=class_names, rotation=45, ha="right")
            plt.yticks(ticks=np.arange(num_classes), labels=class_names)
            plt.colorbar()
            for (i,j), v in np.ndenumerate(cm):
                plt.text(j, i, str(v), ha='center', va='center')
            plt.tight_layout()
            fig.savefig(run_dir / "confusion_matrix.png")
            plt.close(fig)
        except Exception as e:
            print(f"(Aviso) No se pudo guardar PNG de la matriz (matplotlib no disponible?): {e}")

    print("Listo. Mejor modelo:", best_models_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Carpeta con train/ y val/")
    ap.add_argument("--model", default="resnet18", choices=["resnet18","resnet50"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone_lr_scale", type=float, default=0.1,
                    help="Factor para el LR del backbone (ej. 0.1 → cabeza 1e-3, backbone 1e-4)")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--use_pretrained", action="store_true")
    ap.add_argument("--feature_extraction", action="store_true",
                    help="Si se pasa, congela el backbone y entrena solo la cabeza")
    ap.add_argument("--no_feature_extraction", action="store_true",
                    help="Compat: ignora y hace FT completo")
    ap.add_argument("--auto_class_weights", action="store_true",
                    help="Calcula pesos por clase a partir del conteo (útil si hay desbalance)")
    ap.add_argument("--use_rrc", action="store_true",
                    help="Usar RandomResizedCrop en lugar de Resize fijo")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--early_stop", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_confusion", action="store_true",
                    help="Guarda confusion_matrix.csv (+ PNG si hay matplotlib)")
    ap.add_argument("--pin_memory", action="store_true", help="Usar pin_memory en DataLoader (desactivado por defecto)")

    args = ap.parse_args()
    if args.no_feature_extraction:
        args.feature_extraction = False
    train(args)



# python scripts/train_resnet.py --data_dir data/classify --model resnet18 --epochs 20 --batch_size 32 --img_size 384 --use_pretrained --auto_class_weights --backbone_lr_scale 0.1


# python scripts/train_resnet.py --data_dir data/classify --model resnet18 --epochs 20 --batch_size 32 --img_size 384 --use_pretrained --auto_class_weights --backbone_lr_scale 0.1 --workers 0
