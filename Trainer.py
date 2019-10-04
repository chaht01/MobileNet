import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed


class Trainer(object):
    def __init__(
            self,
            loaders,
            model,
            criterion,
            optimizer,
            summarizer,
            device,
            option):
        self.loaders = loaders
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.summarizer = summarizer
        self.option = option
        self.device = device

    def _mode_settings(self, is_train=True):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

    def _train_step(self, epoch, data_loader):
        self._mode_settings(True)
        device = self.device

        for step, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            self.optimizer.zero_grad()
            pred = self.model(images)
            loss = self.criterion(pred, labels)
            loss.backward()
            self.optimizer.step()

    def _validate(self, data_loader):
        self._mode_settings(False)
        device = self.device

        for step, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            pred = self.model(images)
            loss = self.criterion(pred, labels)

    def train(self):
        train_loader, val_loader = self.loaders

        for epoch in range(self.option.epochs):
            self._train_step(epoch, train_loader)
            self._validate(val_loader)
