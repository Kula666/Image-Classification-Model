import torch.nn as nn
import torch.nn.functional as F
import torch


__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding 1"""
    return nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)


class BasicBlock(nn.Module):
    """The number of layers is less than 50"""
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()

        self.features = nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),

            conv3x3(out_planes, out_planes * self.expansion),
            nn.BatchNorm2d(out_planes * self.expansion)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes * self.expansion:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, out_planes * self.expansion, stride),
                nn.BatchNorm2d(out_planes * self.expansion)
            )

    def forward(self, x):
        out = self.features(x)
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class Bottleneck(nn.Module):
    """The number of layers is greater than or equal to 50"""
    expansion = 4

    def __init__(self, in_planes, out_planes, stride):
        super(Bottleneck, self).__init__()

        self.features = nn.Sequential(
            conv1x1(in_planes, out_planes),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),

            conv3x3(out_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),

            conv1x1(out_planes, out_planes * self.expansion),
            nn.BatchNorm2d(out_planes * self.expansion)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes * self.expansion:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, out_planes * self.expansion, stride),
                nn.BatchNorm2d(out_planes * self.expansion)
            )

    def forward(self, x):
        out = self.features(x)
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.features = nn.Sequential(
            conv3x3(3, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            self.__make_layers(block, 16, num_blocks[0], 1),
            self.__make_layers(block, 32, num_blocks[1], 2),
            self.__make_layers(block, 64, num_blocks[2], 2),
            self.__make_layers(block, 128, num_blocks[3], 2)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128 * block.expansion, num_classes)


    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


    def __make_layers(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = list()
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


def resnet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def resnet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def resnet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)

