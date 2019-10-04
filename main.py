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


"""
MobileNet models were trained in TensorFlow [1] using RMSprop [33] with asynchronous gradient descent similar to Inception V3 [31].
However, contrary to training large models we use less regularization and data augmentation techniques because small models have less trouble
with overfitting. When training MobileNets we do not use
side heads or label smoothing and additionally reduce the
amount image of distortions by limiting the size of small
crops that are used in large Inception training [31]. Additionally, we found that it was important to put very little or
no weight decay (l2 regularization) on the depthwise filters
since their are so few parameters in them. For the ImageNet
benchmarks in the next section all models were trained with
same training parameters regardless of the size of the model
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

if option.optimizer == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=option.lr,
                           weight_decay=option.weight_decay)
elif option.optimizer == 'RMSProp':
    optimizer = optim.SGD(net.parameters(), lr=option.lr,
                          momentum=option.momentum, weight_decay=option.weight_decay)
else:  # SGD
    optimizer = optim.SGD(net.parameters(), lr=option.lr,
                          momentum=option.momentum, weight_decay=option.weight_decay)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=10,
    verbose=False,
    threshold=0.0001,
    threshold_mode='rel',
    cooldown=0,
    min_lr=0,
    eps=1e-08)

# Summarizer
summary_path = '%s/runs/%s' % (option.save_dir, option.exp)
os.makedirs(summary_path, exist_ok=True)
summarizer = SummaryWriter(summary_path)

# GPU
device = "cuda:%d" % (option.gpu) if torch.cuda.is_available() else "cpu"

trainer = Trainer(loaders, net, criterion, optimizer, lr_scheduler,
                  summarizer, device, option)

trainer.train()