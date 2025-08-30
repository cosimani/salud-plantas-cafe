# -*- coding: utf-8 -*-
"""
Empaqueta la última corrida en runs/pipeline/<timestamp> y actualiza runs/pipeline/latest.

Origen:
- YOLO   : runs/predict/latest/{annotated/, crops/, crops/crops_manifest.csv, annotated_cls/}
- SAM    : runs/segment/latest/{rgba/, metrics_cutouts.csv}
- Clf    : runs/classify/predict/latest/{predictions.csv, summary_counts.csv, <clases>/*}

Salida (ejemplo):
runs/pipeline/20250829-153012/
  yolo/annotated/*
  yolo/annotated_cls/*
  yolo/crops/*
  yolo/crops/crops_manifest.csv
  sam/rgba/*
  sam/metrics_cutouts.csv
  classify/predictions.csv
  classify/summary_counts.csv
  classify/healthy/*, classify/affected/*
  csv/detections_joined.csv
  csv/summary.csv
  meta.json
y copia atómica a runs/pipeline/latest/
"""

import os, csv, json, shutil, glob
from pathlib import Path
from datetime import datetime

PREDICT_LATEST  = Path("runs/predict/latest")
SEGMENT_LATEST  = Path("runs/segment/latest")
CLASSIFY_LATEST = Path("runs/classify/predict/latest")

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def safe_copytree(src: Path, dst: Path):
    ensure_dir(dst)
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target = dst / (rel if rel != "." else "")
        ensure_dir(target)
        for d in dirs: ensure_dir(target / d)
        for f in files:
            shutil.copy2(os.path.join(root, f), target / f)

def atomic_install_latest(run_root: Path):
    parent = Path("runs/pipeline")
    ensure_dir(parent)
    tmp = parent / ".latest_tmp"
    latest = parent / "latest"
    if tmp.exists():
        if tmp.is_dir(): shutil.rmtree(tmp, ignore_errors=True)
        else: tmp.unlink(missing_ok=True)
    ensure_dir(tmp)
    # copiar
    safe_copytree(run_root, tmp)
    # swap
    if latest.exists():
        if latest.is_dir(): shutil.rmtree(latest, ignore_errors=True)
        else: latest.unlink(missing_ok=True)
    os.replace(tmp, latest)

def read_preds_map(pred_csv: Path):
    """Devuelve {crop_stem: (label, score)} a partir de predictions.csv (RGBA SAM)."""
    m = {}
    if not pred_csv.is_file(): return m
    with open(pred_csv, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            rgba_path = Path(row["file"])
            stem = rgba_path.stem
            cut_pos = stem.rfind("_cut")
            crop_stem = stem[:cut_pos] if cut_pos != -1 else stem
            label = row.get("pred_class","")
            try: score = float(row.get("pred_score","0"))
            except: score = 0.0
            if (crop_stem not in m) or (score > m[crop_stem][1]):
                m[crop_stem] = (label, score)
    return m

def build_join_and_summary(crops_manifest: Path, preds_csv: Path, out_csv_dir: Path):
    ensure_dir(out_csv_dir)
    joined_path  = out_csv_dir / "detections_joined.csv"
    summary_path = out_csv_dir / "summary.csv"
    preds_map = read_preds_map(preds_csv)

    per_image_counts = {}  # {orig_name: {"healthy":n, "affected":n, "total":n}}

    if not crops_manifest.is_file():
        print(f"[WARN] No existe {crops_manifest}")
        return

    with open(crops_manifest, newline="", encoding="utf-8") as fin, \
         open(joined_path, "w", newline="", encoding="utf-8") as fj:

        rd = csv.DictReader(fin)
        headers = [
            "orig_file","crop_file","det_conf",
            "x1","y1","x2","y2","bbox_w","bbox_h","img_w","img_h",
            "pred_class","pred_score"
        ]
        wrj = csv.writer(fj)
        wrj.writerow(headers)

        for row in rd:
            if row.get("status","") != "saved":
                continue
            crop_path = Path(row["crop_file"])
            crop_stem = crop_path.stem
            pred = preds_map.get(crop_stem, ("", 0.0))
            pred_class, pred_score = pred

            # fila join
            wrj.writerow([
                row["orig_file"], row["crop_file"], row.get("conf",""),
                row["x1"], row["y1"], row["x2"], row["y2"],
                row["bbox_w"], row["bbox_h"], row["img_w"], row["img_h"],
                pred_class, f"{pred_score:.6f}" if pred_class else ""
            ])

            # conteos
            name = Path(row["orig_file"]).name
            d = per_image_counts.setdefault(name, {"healthy":0, "affected":0, "total":0})
            if pred_class:
                if pred_class.lower().startswith("heal"): d["healthy"] += 1
                else: d["affected"] += 1
            d["total"] += 1

    # resumen por imagen
    with open(summary_path, "w", newline="", encoding="utf-8") as fs:
        wrs = csv.writer(fs)
        wrs.writerow(["file","total","healthy","affected","pct_healthy","pct_affected"])
        for k, v in sorted(per_image_counts.items()):
            tot = max(v["total"], 1)
            ph = 100.0 * v["healthy"]  / tot
            pa = 100.0 * v["affected"] / tot
            wrs.writerow([k, v["total"], v["healthy"], v["affected"], f"{ph:.2f}", f"{pa:.2f}"])

def main():
    # preparar destino con timestamp
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path("runs/pipeline") / stamp
    yolo_out   = out_root / "yolo"
    sam_out    = out_root / "sam"
    clf_out    = out_root / "classify"
    csv_out    = out_root / "csv"

    # copiar YOLO
    ann = PREDICT_LATEST / "annotated"
    crops = PREDICT_LATEST / "crops"
    ann_cls = PREDICT_LATEST / "annotated_cls"
    if ann.is_dir():     safe_copytree(ann, yolo_out / "annotated")
    if crops.is_dir():   safe_copytree(crops, yolo_out / "crops")
    if ann_cls.is_dir(): safe_copytree(ann_cls, yolo_out / "annotated_cls")

    # copiar SAM
    rgba = SEGMENT_LATEST / "rgba"
    if rgba.is_dir(): safe_copytree(rgba, sam_out / "rgba")
    met_sam = SEGMENT_LATEST / "metrics_cutouts.csv"
    if met_sam.is_file():
        ensure_dir(sam_out)
        shutil.copy2(met_sam, sam_out / "metrics_cutouts.csv")

    # copiar Clasificación (clases + CSVs)
    if CLASSIFY_LATEST.is_dir():
        for p in CLASSIFY_LATEST.iterdir():
            if p.is_dir():
                safe_copytree(p, clf_out / p.name)
            elif p.suffix.lower() == ".csv":
                ensure_dir(clf_out); shutil.copy2(p, clf_out / p.name)

    # CSVs integrados (join + resumen)
    crops_manifest = yolo_out / "crops" / "crops_manifest.csv"
    preds_csv = clf_out / "predictions.csv"
    build_join_and_summary(crops_manifest, preds_csv, csv_out)

    # meta
    meta = {
        "timestamp": stamp,
        "sources": {
            "yolo_latest": str(PREDICT_LATEST.resolve()),
            "sam_latest": str(SEGMENT_LATEST.resolve()),
            "classify_latest": str(CLASSIFY_LATEST.resolve())
        },
        "outputs": {
            "root": str(out_root.resolve()),
            "yolo": str(yolo_out.resolve()),
            "sam": str(sam_out.resolve()),
            "classify": str(clf_out.resolve()),
            "csv": str(csv_out.resolve())
        }
    }
    ensure_dir(out_root)
    with open(out_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # latest atómico
    atomic_install_latest(out_root)

    print("\n[OK] Paquete listo.")
    print(f" - Raíz : {out_root}")
    print(f" - Latest: {Path('runs/pipeline/latest').resolve()}")
    print(f" - CSVs : {csv_out}")

if __name__ == "__main__":
    main()
