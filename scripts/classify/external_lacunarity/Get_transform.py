# -*- coding: utf-8 -*-
"""
Data augmentation & normalization
- CoffeeLeaves / PlantVillage: augment fuerte para robustez (sombras, color, rotación)
- LeavesTex / DeepWeeds: setup similar al original del repo
"""
from __future__ import print_function, division

import random
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# --- Helper: Resize con antialias si tu torchvision lo soporta ---
def ResizeAA(size, interpolation=InterpolationMode.BICUBIC):
    try:
        return transforms.Resize(size, interpolation=interpolation, antialias=True)
    except TypeError:
        # versiones viejas de torchvision no aceptan 'antialias'
        return transforms.Resize(size, interpolation=interpolation)

# --- Compositor para PNG con alfa ---
class RandomAlphaComposite:
    """Si la imagen tiene alfa, la compone sobre un fondo RGB.
       El fondo se elige aleatoriamente de una paleta suave para robustez."""
    def __init__(self, palette=None, fixed=None):
        self.palette = palette or [
            (245, 245, 245), (255, 255, 255),  # grises claros / blanco
            (232, 245, 232), (245, 232, 232),  # verdes/rojizos muy suaves
            (240, 240, 255)                    # azulado suave
        ]
        self.fixed = fixed  # si querés fijar un color fijo, ej. (245,245,245)

    def __call__(self, img):
        if img.mode in ("RGBA", "LA"):
            bg_color = self.fixed or random.choice(self.palette)
            bg = Image.new("RGB", img.size, bg_color)
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            bg.paste(img, mask=img.split()[-1])
            return bg
        # si viene RGB/L, garantizamos RGB
        return img.convert("RGB")

def get_transform(Network_parameters, input_size=224):
    Dataset      = Network_parameters['Dataset']
    resize_size  = Network_parameters.get('resize_size', 256)
    degrees      = Network_parameters.get('degrees', 25)
    use_rotation = Network_parameters.get('rotation', False)

    # --- Estadísticos por dataset ---
    if Dataset == 'LeavesTex':
        mean = [0.3544, 0.4080, 0.1334]
        std  = [0.0312, 0.0344, 0.0638]

    elif Dataset in ['PlantVillage', 'CoffeeLeaves']:
        mean = [0.467, 0.489, 0.412]
        std  = [0.177, 0.152, 0.194]

    elif Dataset == 'DeepWeeds':
        mean = [0.379, 0.39, 0.38]
        std  = [0.224, 0.225, 0.223]

    else:
        raise RuntimeError(f'{Dataset} Dataset not implemented')

    normalize = transforms.Normalize(mean=mean, std=std)

    # === CoffeeLeaves / PlantVillage: imagen pequeña, PNG con alfa, robustez a sombras/color ===
    if Dataset in ['PlantVillage', 'CoffeeLeaves']:
        # train
        train_list = [
            RandomAlphaComposite(),                                  # maneja PNG con alfa
            ResizeAA(384, interpolation=InterpolationMode.BICUBIC),  # preserva detalle
            transforms.RandomCrop(input_size, pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees, fill=(245, 245, 245)),# evita triángulos negros
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            normalize,
        ]
        # test/val
        test_resize = max(resize_size, 384)
        test_list = [
            RandomAlphaComposite(fixed=(245, 245, 245)),
            ResizeAA(test_resize, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]

        return {
            'train': transforms.Compose(train_list),
            'test' : transforms.Compose(test_list),
        }

    # === LeavesTex: estilo original con opción de rotación ===
    if Dataset == 'LeavesTex':
        train_list = [
            transforms.Resize(resize_size),
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        if use_rotation:
            train_list.append(transforms.RandomRotation(degrees, fill=0))
        train_list += [transforms.ToTensor(), normalize]

        test_list = [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]
        return {
            'train': transforms.Compose(train_list),
            'test' : transforms.Compose(test_list),
        }

    # === DeepWeeds: estilo original con opción de rotación ===
    if Dataset == 'DeepWeeds':
        train_list = [
            transforms.Resize(resize_size),
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        if use_rotation:
            train_list.append(transforms.RandomRotation(degrees, fill=0))
        train_list += [transforms.ToTensor(), normalize]

        test_list = [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]
        return {
            'train': transforms.Compose(train_list),
            'test' : transforms.Compose(test_list),
        }
