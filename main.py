import argparse
import random
import os

import torch
import torch.nn as nn
import utils
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets

from models import MobileNet, MobileNet64, BaseLineNet
from option import get_option
from trainer import Trainer
from data_loader import load_data
from torch.utils.tensorboard import SummaryWriter

"""
INCEPTION V3
We have trained our networks with stochastic gradient
utilizing the TensorFlow [1] distributed machine learning
system using 50 replicas running each on a NVidia Kepler
GPU with batch size 32 for 100 epochs. Our earlier experiments used momentum [19] with a decay of 0.9, while our
best models were achieved using RMSProp [21] with decay of 0.9 and  = 1.0. We used a learning rate of 0.045,
decayed every two epoch using an exponential rate of 0.94.
In addition, gradient clipping [14] with threshold 2.0 was
found to be useful to stabilize the training. Model evaluations are performed using a running average of the parameters computed over time.
"""

"""
MOBILENET
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
loaders = load_data(option.dataset, option.workers, option.batch_size)

# Models
if option.arch == 'MobileNet':
    net = MobileNet(width_mult=option.width_mult, shallow=option.shallow)
elif option.arch == 'MobileNet64':
    net = MobileNet64(width_mult=option.width_mult, shallow=option.shallow)
else:
    net = BaseLineNet()

# Criterion, Optimizer, lr_scheduler
criterion = nn.CrossEntropyLoss()

if option.optimizer == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=option.lr,
                           weight_decay=option.weight_decay)
elif option.optimizer == 'RMSProp':
    optimizer = optim.RMSprop(net.parameters(), lr=option.lr,
                              momentum=option.momentum, weight_decay=option.weight_decay, eps=1.0, alpha=0.9)
else:  # SGD
    optimizer = optim.SGD(net.parameters(), lr=option.lr,
                          momentum=option.momentum, weight_decay=option.weight_decay)

if option.lr_scheduler == 'exp':
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=option.lr_decay)
elif option.lr_scheduler == 'step':
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: option.lr_decay ** (epoch // option.lr_step))
elif option.lr_scheduler == 'plat':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=option.lr_patience, factor=option.lr_plat_factor, threshold=0.05)
else:
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: 1.0)  # no decay

# Summarizerr
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

trainer = Trainer(loaders, net, criterion, optimizer, lr_scheduler,
                  summarizer, device, option)

trainer.train()
