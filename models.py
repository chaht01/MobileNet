import torch
import torch.nn as nn
# standard conv -> depthwise conv, pointwise conv(1x1)


class MobileNet(nn.Module):
    def __init__(self, width_mult=1.0, res_mult=1.0, shallow=False):
        super(MobileNet, self).__init__()
        self.width_mult = width_mult
        self.res_mult = res_mult
        self.conv1 = self._conv(3, self._apply_mult(32), 3, 2)
        if shallow is False:
            self.channels = [32, 64, 128, 128, 256,
                             256, 512, 512, 512, 512, 512, 512, 1024, 1024]
            self.stride = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]
        else:
            self.channels = [32, 64, 128, 128, 256,
                             256, 512, 1024, 1024]
            self.stride = [1, 2, 1, 2, 1, 2, 2, 1]
        self.dw = nn.Sequential(
            *[self._dw(self._apply_mult(channel),
                       self._apply_mult(self.channels[i+1]),
                       self.stride[i])
              for i, channel in enumerate(self.channels[:-1])]
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self._apply_mult(self.channels[-1]), 1000)

    def _apply_mult(self, channel):
        return int((channel * self.width_mult) // 1)

    def _dw(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3,
                      stride, 1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.ReLU()
        )

    def _conv(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MobileNe64(nn.Module):
    def __init__(self, width_mult=1.0, res_mult=1.0, shallow=False):
        super(MobileNet64, self).__init__()
        self.width_mult = width_mult
        self.res_mult = res_mult
        self.conv1 = self._conv(3, self._apply_mult(32), 3, 1)
        if shallow is False:
            self.channels = [32, 64, 128, 128, 256,
                             256, 512, 512, 512, 512, 512, 512, 1024, 1024]
            self.stride = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]
        else:
            self.channels = [32, 64, 128, 128, 256,
                             256, 512, 1024, 1024]
            self.stride = [1, 2, 1, 2, 1, 2, 2, 1]
        self.dw = nn.Sequential(
            *[self._dw(self._apply_mult(channel),
                       self._apply_mult(self.channels[i+1]),
                       self.stride[i])
              for i, channel in enumerate(self.channels[:-1])]
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self._apply_mult(self.channels[-1]), 1000)

    def _apply_mult(self, channel):
        return int((channel * self.width_mult) // 1)

    def _dw(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3,
                      stride, 1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.ReLU()
        )

    def _conv(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BaseLineNet(nn.Module):
    def __init__(self):
        super(BaseLineNet, self).__init__()
        self.conv = nn.Sequential(
            self._conv(3, 32, 3, 2, 1),
            self._conv(32, 64, 3, 1, 0),
            self._conv(64, 128, 3, 1, 1),
            self._conv(128, 128, 3, 1, 0),
            self._conv(128, 256, 3, 1, 1),
            self._conv(256, 256, 3, 1, 0),
            self._conv(256, 512, 3, 1, 1),
            self._conv(512, 512, 3, 1, 0),
            self._conv(512, 512, 3, 1, 0),
            self._conv(512, 512, 3, 1, 0),
            self._conv(512, 512, 3, 1, 0),
            self._conv(512, 512, 3, 1, 0),
            self._conv(512, 1024, 3, 1, 1),
            self._conv(1024, 1024, 3, 1, 0)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, 1000)

    def _conv(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
