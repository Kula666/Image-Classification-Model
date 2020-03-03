import torch
import torch.nn as nn


__all__ = ["inception_v1"]


def basic_conv2d(in_planes, out_planes, k_size):
    """Fundamental convolution with BN and ReLU"""
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, k_size, 1, k_size // 2),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


class InceptionV1Module(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(InceptionV1Module, self).__init__()
        self.conv1x1 = basic_conv2d(in_planes, n1x1, 1)
        self.conv3x3 = nn.Sequential(
            basic_conv2d(in_planes, n3x3red, 1),
            basic_conv2d(n3x3red, n3x3, 3)
        )
        self.conv5x5 = nn.Sequential(
            basic_conv2d(in_planes, n5x5red, 1),
            basic_conv2d(n5x5red, n5x5, 5)
        )
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            basic_conv2d(in_planes, pool_planes, 1)
        )

    def forward(self, x):
        y1 = self.conv1x1(x)
        y2 = self.conv3x3(x)
        y3 = self.conv5x5(x)
        y4 = self.maxpool(x)
        return torch.cat([y1, y2, y3, y4], 1)


class InceptionV1(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV1, self).__init__()
        self.pre_layers = basic_conv2d(3, 192, 3)
        self.layer3 = nn.Sequential(
            InceptionV1Module(192, 64, 96, 128, 16, 32, 32),
            InceptionV1Module(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer4 = nn.Sequential(
            InceptionV1Module(480, 192, 96, 208, 16, 48, 64),
            InceptionV1Module(512, 160, 112, 224, 24, 64, 64),
            InceptionV1Module(512, 128, 128, 256, 24, 64, 64),
            InceptionV1Module(512, 112, 114, 288, 32, 64, 64),
            InceptionV1Module(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer5 = nn.Sequential(
            InceptionV1Module(832, 256, 160, 320, 32, 128, 128),
            InceptionV1Module(832, 384, 192, 384, 48, 128, 128)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifer = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifer(x)
        return x


def inception_v1(num_classes):
    return InceptionV1(num_classes)
