import argparse
import torch
import torch.nn as nn
import utils
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets

from models import MobileNet, BaseLineNet
from option import get_option
from Trainer import Trainer
from data_loader import tiny_imagenet
from tensorboardX import SummaryWriter

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

"""
Args
"""


option = get_option()

# dataLoader(option.data_dir)


x = torch.randn((64, 3, 64, 64))


model = MobileNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.2,
                      momentum=0.9, weight_decay=0.9)

baseline = BaseLineNet()


loaders = tiny_imagenet()

Trainer(loaders, model, criterion, optimizer, )

print(model(x).size())
