# scripts/train_yolov8.py
# -*- coding: utf-8 -*-
"""
Entrena YOLOv8 para hojas de café.
Uso: python scripts/train_yolov8.py
- Por defecto usa GPU si está disponible, si no CPU.
- Resultados en runs/detect/yolov8_leaves/
- Copia best.pt en models/best.pt y en models/<timestamp>.pt
"""

import argparse
import os
import sys
import shutil
from datetime import datetime

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    print("ERROR: faltan dependencias. Instalá:\n"
          "  pip install -r requirements-gpu.txt   # o requirements-cpu.txt")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 training runner")

    # Defaults ya configurados
    p.add_argument("--model", default="yolov8s.pt", help="Checkpoint base")
    p.add_argument("--data", default="configs/labels.yaml", help="Dataset YAML")
    p.add_argument("--project", default="runs/detect", help="Carpeta de salidas")
    p.add_argument("--name", default="yolov8_leaves", help="Nombre del experimento")

    # Hiperparámetros
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)

    # Device: auto GPU/CPU
    p.add_argument("--device", default="auto", help="auto / cpu / 0 / 0,1 ...")

    return p.parse_args()


def main():
    args = parse_args()

    # Resolver device automático
    if args.device == "auto":
        args.device = "0" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.project, exist_ok=True)

    model = YOLO(args.model)

    print(f"\n=== Entrenando YOLOv8 ===")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch: {args.batch}")
    print(f"Device: {args.device}")
    print(f"Output dir: {os.path.join(args.project, args.name)}")
    print("================================\n")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True
    )

    # Paths
    out_dir = os.path.join(args.project, args.name, "weights")
    best_path = os.path.join(out_dir, "best.pt")

    if os.path.exists(best_path):
        os.makedirs("models", exist_ok=True)

        # Copia fija como best.pt
        dest_best = os.path.join("models", "best.pt")
        shutil.copy(best_path, dest_best)

        # Copia con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        dest_timestamp = os.path.join("models", f"{timestamp}.pt")
        shutil.copy(best_path, dest_timestamp)

        print("\nEntrenamiento finalizado.")
        print(f"> Copia guardada en: {dest_best}")
        print(f"> Copia con timestamp: {dest_timestamp}")
    else:
        print("⚠️ No se encontró best.pt. ¿El entrenamiento terminó correctamente?")


if __name__ == "__main__":
    main()
