# # -*- coding: utf-8 -*-
# """
# Create datasets and dataloaders for models
# """

# ## Python standard libraries
# from __future__ import print_function
# from __future__ import division
# import pdb
# from Datasets.Split_Data import DataSplit
# import ssl
# ## PyTorch dependencies
# import torch
# ## Local external libraries
# from Datasets.Pytorch_Datasets import *
# from Datasets.Get_transform import *
# from barbar import Bar

# def Compute_Mean_STD(trainloader):
#     print('Computing Mean/STD')
#     'Code from: https://stackoverflow.com/questions/60101240/finding-mean-and-standard-deviation-across-image-channels-pytorch'
#     nimages = 0
#     mean = 0.0
#     var = 0.0
#     for i_batch, batch_target in enumerate(Bar(trainloader)):
#         batch = batch_target[0]
#         # Rearrange batch to be the shape of [B, C, W * H]
#         batch = batch.view(batch.size(0), batch.size(1), -1)
#         # Update total number of images
#         nimages += batch.size(0)
#         # Compute mean and std here
#         mean += batch.mean(2).sum(0) 
#         var += batch.var(2).sum(0)
   
#     mean /= nimages
#     var /= nimages
#     std = torch.sqrt(var)
#     print()
    
#     return mean, std


# def Prepare_DataLoaders(Network_parameters, split):
#     ssl._create_default_https_context = ssl._create_unverified_context
    
#     Dataset = Network_parameters['Dataset']
#     data_dir = Network_parameters['data_dir']    
#     global data_transforms
#     data_transforms = get_transform(Network_parameters, input_size=224)


#     if Dataset == "LeavesTex":
#         train_dataset = LeavesTex1200(data_dir,transform=data_transforms["train"])
#         val_dataset = LeavesTex1200(data_dir,transform=data_transforms["test"])
#         test_dataset = LeavesTex1200(data_dir,transform=data_transforms["test"])
    
#          #Create train/val/test loader
#         split = DataSplit(train_dataset,val_dataset,test_dataset, shuffle=False,random_seed=split)
#         train_loader, val_loader , test_loader = split.get_split(batch_size=Network_parameters['batch_size']['train'], 
#                                                                 num_workers=Network_parameters['num_workers'],
#                                                                 show_sample=False,
#                                                                 val_batch_size=Network_parameters['batch_size']['val'],
#                                                                 test_batch_size=Network_parameters['batch_size']['test'])
#         dataloaders_dict = {'train': train_loader,'val': val_loader,'test': test_loader}



    
#     elif Dataset == "PlantVillage":
#         train_dataset = PlantVillage(data_dir,transform=data_transforms["train"])
#         val_dataset = PlantVillage(data_dir,transform=data_transforms["test"])
#         test_dataset = PlantVillage(data_dir,transform=data_transforms["test"])
    
#          #Create train/val/test loader based on mean and std
#         split = DataSplit(train_dataset,val_dataset,test_dataset, shuffle=False,random_seed=split)
#         train_loader, val_loader , test_loader = split.get_split(batch_size=Network_parameters['batch_size']['train'], 
#                                                                 num_workers=Network_parameters['num_workers'],
#                                                                 show_sample=False,
#                                                                 val_batch_size=Network_parameters['batch_size']['val'],
#                                                                 test_batch_size=Network_parameters['batch_size']['test'])
#         dataloaders_dict = {'train': train_loader,'val': val_loader,'test': test_loader}



#     elif Dataset == "DeepWeeds":
#         train_dataset = DeepWeeds(data_dir,transform=data_transforms["train"])
#         val_dataset = DeepWeeds(data_dir,transform=data_transforms["test"])
#         test_dataset = DeepWeeds(data_dir,transform=data_transforms["test"])
    
#          #Create train/val/test loader based on mean and std
#         split = DataSplit(train_dataset,val_dataset,test_dataset, shuffle=False,random_seed=split)
#         train_loader, val_loader , test_loader = split.get_split(batch_size=Network_parameters['batch_size']['train'], 
#                                                                 num_workers=Network_parameters['num_workers'],
#                                                                 show_sample=False,
#                                                                 val_batch_size=Network_parameters['batch_size']['val'],
#                                                                 test_batch_size=Network_parameters['batch_size']['test'])
#         dataloaders_dict = {'train': train_loader,'val': val_loader,'test': test_loader}

#     elif Params["dataset_name"].lower() == "coffeeleaves":
#         data_dir = os.path.join("Datasets", "CoffeeLeaves")
#         train_dir = os.path.join(data_dir, "train")
#         val_dir   = os.path.join(data_dir, "val")

#         train_dataset = ImageFolderDataset(train_dir, transform=data_transforms["train"])
#         val_dataset   = ImageFolderDataset(val_dir,   transform=data_transforms["val"])

#         train_loader = torch.utils.data.DataLoader(
#             train_dataset, batch_size=Params["batch_size"], shuffle=True,
#             num_workers=Params.get("num_workers", 0)
#         )
#         val_loader = torch.utils.data.DataLoader(
#             val_dataset, batch_size=Params["batch_size"], shuffle=False,
#             num_workers=Params.get("num_workers", 0)
#         )



#     else:
#         raise RuntimeError('{} Dataset not implemented'.format(Dataset)) 
    



#     if Dataset=='LeavesTex' or Dataset=='DeepWeeds':
#             pass
    
#     else:
#         image_datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
#         dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
#                                                         batch_size=Network_parameters['batch_size'][x],
#                                                         num_workers=Network_parameters['num_workers'],
#                                                         pin_memory=Network_parameters['pin_memory'],
#                                                         shuffle=False,
#                                                         )
#                                                         for x in ['train', 'val','test']}
                                                        


#     return dataloaders_dict




# -*- coding: utf-8 -*-
"""
Create datasets and dataloaders for models
"""
from __future__ import print_function, division
import os   # 👈 FALTA
import ssl
import pdb

import torch
from barbar import Bar

from Datasets.Split_Data import DataSplit
from Datasets.Pytorch_Datasets import *
from Datasets.Get_transform import *

def Compute_Mean_STD(trainloader):
    print('Computing Mean/STD')
    nimages, mean, var = 0, 0.0, 0.0
    for i_batch, batch_target in enumerate(Bar(trainloader)):
        batch = batch_target[0].view(batch_target[0].size(0), batch_target[0].size(1), -1)
        nimages += batch.size(0)
        mean += batch.mean(2).sum(0)
        var  += batch.var(2).sum(0)
    mean /= nimages; var /= nimages
    std = torch.sqrt(var)
    print()
    return mean, std

def Prepare_DataLoaders(Network_parameters, split):
    ssl._create_default_https_context = ssl._create_unverified_context

    Dataset  = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']
    global data_transforms
    data_transforms = get_transform(Network_parameters, input_size=224)

    if Dataset == "LeavesTex":
        train_dataset = LeavesTex1200(data_dir, transform=data_transforms["train"])
        val_dataset   = LeavesTex1200(data_dir, transform=data_transforms["test"])
        test_dataset  = LeavesTex1200(data_dir, transform=data_transforms["test"])

        split = DataSplit(train_dataset, val_dataset, test_dataset, shuffle=False, random_seed=split)
        train_loader, val_loader, test_loader = split.get_split(
            batch_size=Network_parameters['batch_size']['train'],
            num_workers=Network_parameters['num_workers'],
            show_sample=False,
            val_batch_size=Network_parameters['batch_size']['val'],
            test_batch_size=Network_parameters['batch_size']['test'])
        dataloaders_dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    elif Dataset == "PlantVillage":
        train_dataset = PlantVillage(data_dir, transform=data_transforms["train"])
        val_dataset   = PlantVillage(data_dir, transform=data_transforms["test"])
        test_dataset  = PlantVillage(data_dir, transform=data_transforms["test"])

        split = DataSplit(train_dataset, val_dataset, test_dataset, shuffle=False, random_seed=split)
        train_loader, val_loader, test_loader = split.get_split(
            batch_size=Network_parameters['batch_size']['train'],
            num_workers=Network_parameters['num_workers'],
            show_sample=False,
            val_batch_size=Network_parameters['batch_size']['val'],
            test_batch_size=Network_parameters['batch_size']['test'])
        dataloaders_dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    elif Dataset == "DeepWeeds":
        train_dataset = DeepWeeds(data_dir, transform=data_transforms["train"])
        val_dataset   = DeepWeeds(data_dir, transform=data_transforms["test"])
        test_dataset  = DeepWeeds(data_dir, transform=data_transforms["test"])

        split = DataSplit(train_dataset, val_dataset, test_dataset, shuffle=False, random_seed=split)
        train_loader, val_loader, test_loader = split.get_split(
            batch_size=Network_parameters['batch_size']['train'],
            num_workers=Network_parameters['num_workers'],
            show_sample=False,
            val_batch_size=Network_parameters['batch_size']['val'],
            test_batch_size=Network_parameters['batch_size']['test'])
        dataloaders_dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    # ✅ Nuevo branch: CoffeeLeaves (ImageFolder con train/val/(test opcional))
    elif Dataset == "CoffeeLeaves":
        train_dir = os.path.join(data_dir, "train")
        val_dir   = os.path.join(data_dir, "val")
        test_dir  = os.path.join(data_dir, "test")  # si no existe, usamos val como test

        if not os.path.isdir(test_dir):
            test_dir = val_dir

        # Usa la clase que agregaste en Pytorch_Datasets.py
        train_dataset = ImageFolderDataset(train_dir, transform=data_transforms["train"])
        val_dataset   = ImageFolderDataset(val_dir,   transform=data_transforms["test"])
        test_dataset  = ImageFolderDataset(test_dir,  transform=data_transforms["test"])

    else:
        raise RuntimeError('{} Dataset not implemented'.format(Dataset))

    # 👇 Este bloque arma los DataLoaders para todo lo que NO sea LeavesTex/DeepWeeds
    if Dataset == 'LeavesTex' or Dataset == 'DeepWeeds':
        pass
    else:
        image_datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
        dataloaders_dict = {
            x: torch.utils.data.DataLoader(
                image_datasets[x],
                batch_size=Network_parameters['batch_size'][x],
                num_workers=Network_parameters['num_workers'],
                pin_memory=Network_parameters['pin_memory'],
                shuffle=(x == 'train'),  # 👈 train con shuffle
            )
            for x in ['train', 'val', 'test']
        }

    return dataloaders_dict
    