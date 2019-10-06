import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.datasets as datasets

import numpy as np
from PIL import Image
import os
from data.stanford_dogs import dogs


def load_data(data='tiny-imagenet', workers=4, batch_size=128):
    if data == 'tiny-imagenet':
        train_dataset, test_dataset, classes = tiny_imagenet(
            '../../data/tiny-imagenet-200')
    elif data == 'stanford-dogs':
        train_dataset, test_dataset, classes = stanford_dogs('../../data')

    tr_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    te_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    return tr_loader, te_loader


def tiny_imagenet(data_dir, workers=4, batch_size=64):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize, ])

    train_dataset = datasets.ImageFolder(
        data_dir+'/train', transform=transform_train)

    classes = train_dataset.classes

    transform_test = transforms.Compose([
        transforms.ToTensor(), normalize, ])

    test_dataset = datasets.ImageFolder(
        data_dir+'/val', transform=transform_test)

    return train_dataset, test_dataset, classes


def stanford_dogs(data_dir, workers=4, batch_size=64):
    """
    https://github.com/zrsmithson/Stanford-dogs/blob/master/data/load.py
    """
    input_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, ratio=(1, 1.3)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    train_dataset = dogs(root=data_dir,
                         train=True,
                         cropped=False,
                         transform=input_transforms,
                         download=False)
    test_dataset = dogs(root=data_dir,
                        train=False,
                        cropped=False,
                        transform=input_transforms,
                        download=False)

    classes = train_dataset.classes

    return train_dataset, test_dataset, classes
