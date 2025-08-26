# -*- coding: utf-8 -*-
import argparse, os, glob, shutil, time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms, datasets

from Demo_Parameters import Parameters as DemoParameters
from Prepare_Data import Prepare_DataLoaders
from Datasets.Get_transform import get_transform
from Utils.Network_functions import initialize_model

def find_latest_weights(params):
    """
    Busca el último Best_Weights.pt bajo Saved_Models/.../CoffeeLeaves/.../Run_*/Best_Weights.pt
    usando la estructura que guarda Save_Results.py
    """
    folder = params['folder']               # p.ej. 'Saved_Models'
    pooling = params['pooling_layer']       # p.ej. 'Base_Lacunarity'
    agg     = params['agg_func']            # p.ej. 'global'
    mode    = params['mode']                # 'Feature_Extraction' o 'Fine_Tuning'
    dataset = params['Dataset']             # 'CoffeeLeaves'
    model   = params['Model_name']          # p.ej. 'resnet18'

    pattern = os.path.join(folder, pooling, agg, mode, dataset, model, "Run_*", "Best_Weights.pt")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No se encontró Best_Weights.pt con patrón:\n{pattern}")
    # el más reciente por mtime
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]

def build_params_for_inference():
    """
    Replica los parámetros por defecto de demo.py pero fijando dataset CoffeeLeaves.
    Si cambiaste algo al entrenar (pooling, agg, backbone, etc.), podés cambiarlo acá o
    pasar los mismos flags que en demo.py cuando entrenaste.
    """
    class A:  # simple Namespace
        save_results = False
        xai = False
        earlystoppping = 10
        folder = 'Saved_Models'
        
        pooling_layer = 5          # 5 -> 'Base_Lacunarity'
        pooling_layer = 6          # 6 -> 'MS_Lacunarity'  ✅ para esos pesos
        agg_func = 1               # 1 = 'global'
        model = 'resnet18'

        data_selection = 4         # 4 -> 'CoffeeLeaves' (lo agregaste en Demo_Parameters.py)
        kernel = None
        stride = None
        padding = 0
        num_levels = 2
        feature_extraction = True
        use_pretrained = True
        lr = 0.01
        train_batch_size = 16
        val_batch_size = 16
        test_batch_size = 16
        num_epochs = 1
        resize_size = 256
        model = 'resnet18'
        use_cuda = True
    return DemoParameters(A)

def load_model(params, dataloaders_dict, weights_path, device):
    num_classes = params['num_classes'][params['Dataset']]
    model_name  = params['Model_name']

    model, _ = initialize_model(model_name, num_classes, dataloaders_dict, params,
                                aggFunc=params["agg_func"])
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True,  help="Carpeta con imágenes a clasificar")
    ap.add_argument("--output", default="classified_output", help="Carpeta de salida")
    ap.add_argument("--weights", default=None, help="Ruta a Best_Weights.pt (opcional: si no, se busca el último)")
    ap.add_argument("--move", action="store_true", help="Mover en lugar de copiar")
    args = ap.parse_args()

    # 1) Params + device
    params = build_params_for_inference()
    device = torch.device("cuda:0" if torch.cuda.is_available() and params.get('use_cuda', True) else "cpu")

    # 2) Dataloaders dummy (para inicializar el modelo con la misma config)
    #    Usamos el conjunto 'val' de CoffeeLeaves (ya existente) solo para construir el modelo
    data_transforms = get_transform(params, input_size=224)
    coffee_dir = params['data_dir']  # 'Datasets/CoffeeLeaves'
    val_dir = os.path.join(coffee_dir, "val")
    if not os.path.isdir(val_dir):
        raise RuntimeError(f"No existe {val_dir}. Asegurate de tener Datasets/CoffeeLeaves/val/...")
    dummy_val = datasets.ImageFolder(root=val_dir, transform=data_transforms["test"])
    dummy_loader = torch.utils.data.DataLoader(dummy_val, batch_size=1, shuffle=False, num_workers=0)
    dataloaders_dict = {"train": dummy_loader, "val": dummy_loader, "test": dummy_loader}

    # 3) Pesos
    weights_path = args.weights or find_latest_weights(params)
    print(f"[INFO] Usando pesos: {weights_path}")

    # 4) Cargar modelo
    model = load_model(params, dataloaders_dict, weights_path, device)

    # 5) Mapeo de clases (tomamos el orden del entrenamiento)
    #    Usamos el mapping de ImageFolder del train para garantizar el mismo orden.
    train_dir = os.path.join(coffee_dir, "train")
    train_ds_for_classes = datasets.ImageFolder(root=train_dir)
    idx_to_class = {v: k for k, v in train_ds_for_classes.class_to_idx.items()}
    print(f"[INFO] Clases: {idx_to_class}")

    # 6) Transform de inferencia (igual a test)
    infer_tf = data_transforms["test"]

    # 7) I/O
    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cname in idx_to_class.values():
        (out_dir / cname).mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = [p for p in in_dir.rglob("*") if p.suffix.lower() in exts]
    if not images:
        print(f"[WARN] No encontré imágenes en {in_dir}")
        return

    print(f"[INFO] Clasificando {len(images)} imágenes desde {in_dir} ...")
    t0 = time.time()
    for p in images:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"[SKIP] {p} ({e})")
            continue

        x = infer_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            pred = int(torch.argmax(logits, dim=1).item())
        pred_name = idx_to_class[pred]

        dest = out_dir / pred_name / p.name
        if args.move:
            shutil.move(str(p), str(dest))
        else:
            shutil.copy2(str(p), str(dest))
    dt = time.time() - t0
    print(f"[DONE] Listo. Salida en: {out_dir}  (tiempo {dt:.1f}s)")

if __name__ == "__main__":
    main()



# Ejecutar con:
# python classify_folder.py --input "C:\Cosas\2025\IBERO2025\artículos IBERO ago 2025\2024_V4A_Lacunarity_Pooling_Layer-main\Datasets\CoffeeLeaves\sin_clasificar" --output "C:\Cosas\2025\IBERO2025\artículos IBERO ago 2025\2024_V4A_Lacunarity_Pooling_Layer-main\Datasets\CoffeeLeaves\clasificadas" --move

# python classify_folder.py --input "C:\Cosas\2025\IBERO2025\artículos IBERO ago 2025\2024_V4A_Lacunarity_Pooling_Layer-main\Datasets\CoffeeLeaves\sin_clasificar" --output "C:\Cosas\2025\IBERO2025\artículos IBERO ago 2025\2024_V4A_Lacunarity_Pooling_Layer-main\Datasets\CoffeeLeaves\clasificadas" --weights "C:\Cosas\2025\IBERO2025\artículos IBERO ago 2025\2024_V4A_Lacunarity_Pooling_Layer-main\Saved_Models\MS_Lacunarity\global\Fine_Tuning\CoffeeLeaves\resnet18\Run_1\Best_Weights.pt" --move

           

