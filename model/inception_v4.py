# -*- coding: UTF-8 -*-
""" inceptionv4 in pytorch


[1] Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi

    Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
    https://arxiv.org/abs/1602.07261
"""

import torch
import torch.nn as nn


__all__ = ["inception_v4"]


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, k_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, k_size, bias=False, **kwargs),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.basic_conv(x)


class InceptionStem(nn.Module):
    """
    Figure 3. The schema for stem of the pure Inception-v4
    and Inception-ResNet-v2 networks.
    This is the input part of those net-works.
    """
    def __init__(self, in_planes):
        super(InceptionStem, self).__init__()
        self.conv1 = nn.Sequential(
            BasicConv2d(in_planes, 32, 3),
            BasicConv2d(32, 32, 3, padding=1),
            BasicConv2d(32, 64, 3, padding=1)
        )
        self.branch_3x3_conv = BasicConv2d(64, 96, 3, padding=1)
        self.branch_3x3_pool = nn.MaxPool2d(3, 1, 1)
        self.branch_7x7a = nn.Sequential(
            BasicConv2d(160, 64, 1),
            BasicConv2d(64, 96, 3, padding=1)
        )
        self.branch_7x7b = nn.Sequential(
            BasicConv2d(160, 64, 1),
            BasicConv2d(64, 64, (7, 1), padding=(3, 0)),
            BasicConv2d(64, 64, (1, 7), padding=(0, 3)),
            BasicConv2d(64, 96, 3, padding=1)
        )
        self.branchpoola = nn.MaxPool2d(3, 1, 1)
        self.branchpoolb = BasicConv2d(192, 192, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = [
            self.branch_3x3_conv(x),
            self.branch_3x3_pool(x)
        ]
        x = torch.cat(x, 1)
        x = [
            self.branch_7x7a(x),
            self.branch_7x7b(x)
        ]
        x = torch.cat(x, 1)
        x = [
            self.branchpoola(x),
            self.branchpoolb(x)
        ]
        return torch.cat(x, 1)


class InceptionA(nn.Module):
    """
    Figure 4. The schema for 35 × 35 grid modules of the pure
    Inception-v4 network. This is the Inception-A block of Figure 9.
    """
    def __init__(self, in_planes):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_planes, 96, 1)
        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_planes, 64, 1),
            BasicConv2d(64, 96, 3, padding=1)
        )
        self.branch5x5 = nn.Sequential(
            BasicConv2d(in_planes, 64, 1),
            BasicConv2d(64, 96, 3, padding=1),
            BasicConv2d(96, 96, 3, padding=1)
        )
        self.branchpool = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BasicConv2d(in_planes, 96, 1)
        )

    def forward(self, x):
        x = [
            self.branch1x1(x),
            self.branch3x3(x),
            self.branch5x5(x),
            self.branchpool(x)
        ]
        return torch.cat(x, 1)


class ReductionA(nn.Module):
    """
    Figure 7. The schema for 35 × 35 to 17 × 17 reduction module.
    Different variants of this blocks (with various number of filters)
    are used in Figure 9, and 15 in each of the new Inception(-v4, -
    ResNet-v1, -ResNet-v2) variants presented in this paper. The k, l,
    m, n numbers represent filter bank sizes which can be looked up
    in Table 1.
    """
    def __init__(self, in_planes, k, l, m, n):
        super(ReductionA, self).__init__()
        self.branch3x3 = BasicConv2d(in_planes, n, 3, stride=2)
        self.branch5x5 = nn.Sequential(
            BasicConv2d(in_planes, k, 1),
            BasicConv2d(k, l, 3, padding=1),
            BasicConv2d(l, m, 3, stride=2)
        )
        self.branchpool = nn.MaxPool2d(3, 2)

        self.out_planes = n + m + in_planes

    def forward(self, x):
        x = [
            self.branch3x3(x),
            self.branch5x5(x),
            self.branchpool(x)
        ]
        return torch.cat(x, 1)


class InceptionB(nn.Module):
    """
    Figure 5. The schema for 17 × 17 grid modules of the pure
    Inception-v4 network. This is the Inception-B block of Figure 9.
    """
    def __init__(self, in_planes):
        super(InceptionB, self).__init__()
        self.branch1x1 = BasicConv2d(in_planes, 384, 1)
        self.branch7x7 = nn.Sequential(
            BasicConv2d(in_planes, 192, 1),
            BasicConv2d(192, 224, (1, 7), padding=(0, 3)),
            BasicConv2d(224, 256, (7, 1), padding=(3, 0))
        )
        self.branch7x7stack = nn.Sequential(
            BasicConv2d(in_planes, 192, 1),
            BasicConv2d(192, 192, (1, 7), padding=(0, 3)),
            BasicConv2d(192, 224, (7, 1), padding=(3, 0)),
            BasicConv2d(224, 224, (1, 7), padding=(0, 3)),
            BasicConv2d(224, 256, (7, 1), padding=(3, 0))
        )
        self.branchpool = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BasicConv2d(in_planes, 128, 1)
        )

    def forward(self, x):
        x = [
            self.branch1x1(x),
            self.branch7x7(x),
            self.branch7x7stack(x),
            self.branchpool(x)
        ]
        return torch.cat(x, 1)


class ReductionB(nn.Module):
    """
    Figure 8. The schema for 17 × 17 to 8 × 8 grid-reduction mod-
    ule. This is the reduction module used by the pure Inception-v4
    network in Figure 9
    """
    def __init__(self, in_planes):
        super(ReductionB, self).__init__()
        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_planes, 192, 1),
            BasicConv2d(192, 192, 3, stride=2, padding=1)
        )
        self.branch7x7 = nn.Sequential(
            BasicConv2d(in_planes, 256, 1),
            BasicConv2d(256, 256, (1, 7), padding=(0, 3)),
            BasicConv2d(256, 320, (7, 1), padding=(3, 0)),
            BasicConv2d(320, 320, 3, stride=2, padding=1)
        )
        self.branchpool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        x = [
            self.branch3x3(x),
            self.branch7x7(x),
            self.branchpool(x)
        ]
        return torch.cat(x, 1)


class InceptionC(nn.Module):
    """
    Figure 6. The schema for 8×8 grid modules of the pure Inception-
    v4 network. This is the Inception-C block of Figure 9.
    """
    def __init__(self, in_planes):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_planes, 256, 1)
        self.branch3x3 = BasicConv2d(in_planes, 384, 1)
        self.branch3x3a = BasicConv2d(384, 256, (1, 3), padding=(0, 1))
        self.branch3x3b = BasicConv2d(384, 256, (3, 1), padding=(1, 0))
        self.branch5x5 = nn.Sequential(
            BasicConv2d(in_planes, 384, 1),
            BasicConv2d(384, 448, (1, 3), padding=(0, 1)),
            BasicConv2d(448, 512, (3, 1), padding=(1, 0))
        )
        self.branch5x5a = BasicConv2d(512, 256, (3, 1), padding=(1, 0))
        self.branch5x5b = BasicConv2d(512, 256, (1, 3), padding=(0, 1))
        self.branchpool = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BasicConv2d(in_planes, 256, 1)
        )

    def forward(self, x):
        y1 = self.branch1x1(x)
        y2 = self.branch3x3(x)
        y2 = [
            self.branch3x3a(y2),
            self.branch3x3b(y2)
        ]
        y2 = torch.cat(y2, 1)
        y3 = self.branch5x5(x)
        y3 = [
            self.branch5x5a(y3),
            self.branch5x5b(y3)
        ]
        y3 = torch.cat(y3, 1)
        y4 = self.branchpool(x)
        return torch.cat([y1, y2, y3, y4], 1)


class InceptionV4(nn.Module):
    def __init__(self, A, B, C, k=192, l=224, m=256, n=384, num_classes=10):
        super(InceptionV4, self).__init__()
        self.stem = InceptionStem(3)
        self.inceptionA = self.__make_block(384, 384, A, InceptionA)
        self.reductionA = ReductionA(384, k, l, m, n)
        out_planes = self.reductionA.out_planes
        self.inceptionB = self.__make_block(out_planes, 1024, B, InceptionB)
        self.reductionB = ReductionB(1024)
        self.inceptionC = self.__make_block(1536, 1536, C, InceptionC)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(1 - 0.8)
        self.classifier = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inceptionA(x)
        x = self.reductionA(x)
        x = self.inceptionB(x)
        x = self.reductionB(x)
        x = self.inceptionC(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def __make_block(self, in_planes, out_planes, num_blocks, block):
        layers = list()
        for i in range(num_blocks):
            layers.append(block(in_planes))
            in_planes = out_planes
        return nn.Sequential(*layers)


def inception_v4(num_classes):
    return InceptionV4(4, 7, 3, num_classes=num_classes)
