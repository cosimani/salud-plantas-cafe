# -*- coding: utf-8 -*-
"""
Orquesta el pipeline completo usando los **valores por defecto** de cada etapa:
  1) YOLOv8 (detect + crops + annotated) → actualiza runs/predict/latest
  2) SAM cut-outs desde runs/predict/latest/crops → actualiza runs/segment/latest
  3) Clasificación desde runs/segment/latest/rgba → actualiza runs/classify/predict/latest

Uso:
    python scripts/run_all_defaults.py
Requisitos:
    - models/best.pt
    - models/resnet18_best.pt  (o algún *_best.pt detectable por predict_resnet.py)
    - data/yolo/images/test/   (o tus imágenes en ese path)
"""

import sys
import subprocess
from pathlib import Path

def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    py = sys.executable

    # 1) YOLOv8: detecta, recorta y anota (usa sus defaults)
    run([py, "scripts/detect_and_crop_yolov8.py"])

    # 2) SAM: toma crops de runs/predict/latest/crops (usa sus defaults)
    run([py, "scripts/sam_segment_crops.py"])

    # 3) Clasificación: toma rgba de runs/segment/latest/rgba (usa sus defaults)
    run([py, "scripts/predict_resnet.py"])

    # 4) Anotar originales con healthy/affected según clasificación
    run([py, "scripts/annotate_from_predictions.py"])

    # 5) Empaquetar todo a runs/pipeline/<timestamp> y latest
    run([py, "scripts/collect_pipeline_outputs.py"])

    print("\n[OK] Pipeline completo con parámetros por defecto.")
    print(" - YOLO   → runs/predict/latest/")
    print(" - SAM    → runs/segment/latest/")
    print(" - Clf    → runs/classify/predict/latest/")

if __name__ == "__main__":
    main()
