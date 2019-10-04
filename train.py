import argparse
import torch
import torch.nn as nn
from models import MobileNet, BaseLineNet
import utils
import torch.optim as optim
import torchvision
from torchvision import transforms
import option

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


x = torch.randn((64, 3, 224, 224))


model = MobileNet()
optimizer = optim.RMSprop(model.parameters(), lr=0.2,
                          momentum=0.9, weight_decay=0.9, eps=1.0)

baseline = BaseLineNet()
