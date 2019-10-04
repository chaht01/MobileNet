import torch
import torch.nn as nn


# standard conv -> depthwise conv, pointwise conv(1x1)

class DepthWiseSeparableConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1):
        super(DepthWiseSeparableConv, self).__init__()
        self.in_channels = in_channels

        self.depthWiseConvs = [
            nn.Conv2d(1, 1, 3, stride, 1)
        ] * in_channels
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.pointWiseConvs = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        N, C, _, _ = x.size()
        assert C == self.in_channels, "Expected input channel %d but got %d" % (
            self.in_channels, C)
        depth_wise_layers = []
        for c in range(C):
            depth_wise_layers.append(self.depthWiseConvs[c](x[:, c:c+1, :, :]))

        x = torch.cat(tuple(depth_wise_layers), 1)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pointWiseConvs(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class MobileNet(nn.Module):
    def __init__(self, width_mult=1.0, res_mult=1.0):
        super(MobileNet, self).__init__()
        self.width_mult = width_mult
        self.res_mult = res_mult
        self.conv1 = self._conv(3, 32, 3, 1)
        self.channels = [32, 64, 128, 128, 256,
                         256, 512, 512, 512, 512, 512, 512, 1024, 1024]
        self.stride = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2]
        self.dw = nn.Sequential(
            *[DepthWiseSeparableConv(
              self._apply_mult(channel, self.width_mult),
              self._apply_mult(self.channels[i+1], self.width_mult),
              self.stride[i])
              for i, channel in enumerate(self.channels[:-1])]
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, 1000)

    def _apply_mult(self, channel, mult):
        return int((channel * mult) // 1)

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
        print(x.size())
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
