import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import shutil
import utils
import os
import time
import json


class Trainer(object):
    def __init__(
            self,
            loaders,
            model,
            criterion,
            optimizer,
            lr_scheduler,
            summarizer,
            device,
            option):
        self.option = option
        self.device = device

        os.makedirs('%s/%s' % (self.option.save_dir,
                               self.option.exp), exist_ok=True)

        with open('%s/%s/commandline_args.txt' % (self.option.save_dir, self.option.exp), 'w') as f:
            json.dump(self.option.__dict__, f, indent=2)

        print("Train on %s" % (self.device))
        self.loaders = loaders
        self.model = model
        self.model = self.model.to(device)
        self.criterion = criterion
        self.criterion = self.criterion.to(device)
        self.optimizer = optimizer
        self.summarizer = summarizer
        self.lr_scheduler = lr_scheduler

        self.best_acc = 0

        if self.option.resume is not None:
            state_dict = torch.load(self.option.resume)
            self.model.load_state_dict(state_dict['state_dict'])
            self.option.start_epoch = state_dict['epoch']
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.best_acc = state_dict['best_acc']
            self.lr_scheduler.load_state_dict(state_dict['scheduler'])

    def _set_training(self, is_train=True):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _train_step(self, epoch, data_loader):
        self._set_training(True)
        device = self.device
        losses = utils.AverageMeter()
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()

        end = time.time()
        for step, (images, labels) in enumerate(data_loader):
            # data loading time
            data_time.update(time.time() - end)

            # get loss
            images = images.to(device)
            labels = labels.to(device)
            self.optimizer.zero_grad()
            pred = self.model(images)
            loss = self.criterion(pred, labels)
            loss.backward()
            self.optimizer.step()

            # Compute average Loss
            batch_time.update(time.time() - end)
            end = time.time()
            losses.update(loss.item(), len(images))

            if step % self.option.log_step == 0:
                print("Train: [Epoch %d] [step %d] [loss %.6f] [loading %6.3f] [batch elapse %6.3f]" % (
                    epoch, step, losses.avg, data_time.avg, batch_time.avg))

        # Decay learning rate using plateu policy
        self.summarizer.add_scalar('lr/train', self._get_lr(), epoch)
        self.lr_scheduler.step(losses.avg)
        self.summarizer.add_scalar('loss/train', losses.avg, epoch)

        # self.summarizer.add_scalar('lr', )

    def _validate_step(self, epoch, data_loader):
        self._set_training(False)
        device = self.device
        losses = utils.AverageMeter()
        accuracy = utils.AverageMeter()
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        correct = 0
        cnt = 0

        end = time.time()
        for step, (images, labels) in enumerate(data_loader):
            # data loading time
            data_time.update(time.time() - end)

            # get loss
            images = images.to(device)
            labels = labels.to(device)
            pred = self.model(images)
            loss = self.criterion(pred, labels)

            # Compute average Loss
            batch_time.update(time.time() - end)
            end = time.time()
            losses.update(loss.item(), len(images))

            # Compute average Accuracy
            correct += (pred.argmax(1) == labels).sum().item()
            cnt += len(images)
            accuracy.update(correct / cnt, len(images))

            if step % self.option.log_step == 0:
                print("Valid: [Epoch %d] [step %d] [loss %.6f] [acc %.6f] [loading %6.3f] [batch elapse %6.3f]" % (
                    epoch, step, losses.avg, accuracy.avg, data_time.avg, batch_time.avg))

        self.summarizer.add_scalar('acc/valid', accuracy.avg, epoch)
        self.summarizer.add_scalar('loss/valid', losses.avg, epoch)

        return accuracy.avg

    def train(self):
        train_loader, val_loader = self.loaders

        is_best = False
        best_epoch = 0
        print("="*30)
        print(self.option)
        print("="*30)

        for epoch in range(self.option.start_epoch+1, self.option.epochs):
            filename = '%s/%s/%d.pth' % (self.option.save_dir,
                                      self.option.exp, epoch)

            self._train_step(epoch, train_loader)
            acc = self._validate_step(epoch, val_loader)

            if self.best_acc < acc:
                self.best_acc = acc
                is_best = True

            torch.save({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'best_acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.lr_scheduler.state_dict()
            }, filename)

            if is_best:
                shutil.copyfile(filename, '%s/%s/best.pth' %
                                (self.option.save_dir, self.option.exp))
                best_epoch = epoch

            is_best = False

        print("Best model [acc: %.6f] at [epoch %d]" %
              (self.best_acc, best_epoch))
        self.summarizer.close()
