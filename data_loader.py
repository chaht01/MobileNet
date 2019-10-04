import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.datasets as datasets

import numpy as np
from PIL import Image
import os


def tiny_imagenet(data_dir, workers=4, batch_size=64):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize, ])
    tr_dataset = datasets.ImageFolder(
        data_dir+'/train', transform=transform_train)
    tr_loader = DataLoader(
        tr_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    # evaluation during training
    transform_test = transforms.Compose([
        transforms.ToTensor(), normalize, ])
    te_dataset = datasets.ImageFolder(
        data_dir+'/val', transform=transform_test)
    te_loader = DataLoader(
        te_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    return tr_loader, te_loader
