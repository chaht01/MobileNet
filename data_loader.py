import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.datasets as datasets

import numpy as np
from PIL import Image
import os


class SomeDataLoader(Dataset):
    def __init__(self, option):
        super(SomeDataLoader, self).__init__()
        # extract pre defined options for data loading
        data_dir = option.data_dir
        workers = option.num_workers
        self.train_transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
