import argparse
import random
import os

import torch
import torch.nn as nn
import utils
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets

from models import MobileNet, BaseLineNet
from option import get_option
from trainer import Trainer
from data_loader import tiny_imagenet
from torch.utils.tensorboard import SummaryWriter
from utils import AverageMeter
import math

"""
https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee
"""
seed = 2
random.seed(seed)
torch.manual_seed(seed)


# Get option(args)
option = get_option()

# Loaders
loaders = tiny_imagenet(option.data_dir, option.workers, option.batch_size)

# Models
if option.arch == 'MobileNet':
    net = MobileNet(width_mult=option.width_mult)
else:
    net = BaseLineNet()

# Criterion, Optimizer, lr_scheduler
criterion = nn.CrossEntropyLoss()


end_lr = 1
start_lr = 1e-7
lr_find_epochs = 2
optimizer = optim.SGD(net.parameters(), lr=start_lr)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda x: math.exp(x * math.log(end_lr / start_lr) / (lr_find_epochs * len(loaders[0]))))

# Summarizer
summary_path = '%s/runs/%s' % (option.save_dir, option.exp)
os.makedirs(summary_path, exist_ok=True)
if option.resume is not None:
    state_dict = torch.load(option.resume)
    purge_step = state_dict['epoch']+1
    summarizer = SummaryWriter(summary_path, purge_step=purge_step)
else:
    summarizer = SummaryWriter(summary_path)

# GPU
device = "cuda:%d" % (option.gpu) if torch.cuda.is_available() else "cpu"

# average loss
net.to(device)
for i in range(lr_find_epochs):
    for step, (inputs, labels) in enumerate(loaders[0]):
        # Send to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Training mode and zero gradients
        net.train()
        optimizer.zero_grad()

        # Get outputs to calc loss
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update LR
        lr_scheduler.step()
        lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
        summarizer.add_scalar('lr', lr_step, i*len(inputs) + step)
        summarizer.add_scalar('loss', loss.item(), i*len(inputs) + step)
