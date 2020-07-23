import torch
import torch.nn as nn


__all__ = ["inception_v3"]


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


class InceptionA(nn.Module):
    def __init__(self, in_planes, pool_planes):
        super(InceptionA, self).__init__()

        self.branch1x1 = BasicConv2d(in_planes, 64, 1)
        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_planes, 64, 1),
            BasicConv2d(64, 96, 3, padding=1),
            BasicConv2d(96, 96, 3, padding=1)
        )
        self.branch5x5 = nn.Sequential(
            BasicConv2d(in_planes, 48, 1),
            BasicConv2d(48, 64, 5, padding=2)
        )
        self.branchpool = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BasicConv2d(in_planes, pool_planes, 1)
        )

    def forward(self, x):
        y1 = self.branch1x1(x)
        y2 = self.branch3x3(x)
        y3 = self.branch5x5(x)
        y4 = self.branchpool(x)
        return torch.cat([y1, y2, y3, y4], 1)


class InceptionB(nn.Module):
    def __init__(self, in_planes):
        super(InceptionB, self).__init__()

        self.branch3x3 = BasicConv2d(in_planes, 384, 3, stride=2)
        self.branch3x3stack = nn.Sequential(
            BasicConv2d(in_planes, 64, 1),
            BasicConv2d(64, 96, 3, padding=1),
            BasicConv2d(96, 96, 3, stride=2)
        )
        self.branchpool = nn.MaxPool2d(3, 2)

    def forward(self, x):
        y1 = self.branch3x3(x)
        y2 = self.branch3x3stack(x)
        y3 = self.branchpool(x)
        return torch.cat([y1, y2, y3], 1)


class InceptionC(nn.Module):
    def __init__(self, in_planes, c7x7):
        super(InceptionC, self).__init__()

        self.branch1x1 = BasicConv2d(in_planes, 192, 1)
        self.branch7x7 = nn.Sequential(
            BasicConv2d(in_planes, c7x7, 1),
            BasicConv2d(c7x7, c7x7, (1, 7), padding=(0, 3)),
            BasicConv2d(c7x7, 192, (7, 1), padding=(3, 0))
        )
        self.branch7x7stack = nn.Sequential(
            BasicConv2d(in_planes, c7x7, 1),
            BasicConv2d(c7x7, c7x7, (1, 7), padding=(0, 3)),
            BasicConv2d(c7x7, c7x7, (7, 1), padding=(3, 0)),
            BasicConv2d(c7x7, c7x7, (1, 7), padding=(0, 3)),
            BasicConv2d(c7x7, 192, (7, 1), padding=(3, 0))
        )
        self.branchpool = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BasicConv2d(in_planes, 192, 1)
        )

    def forward(self, x):
        y1 = self.branch1x1(x)
        y2 = self.branch7x7(x)
        y3 = self.branch7x7stack(x)
        y4 = self.branchpool(x)
        return torch.cat([y1, y2, y3, y4], 1)


class InceptionD(nn.Module):
    def __init__(self, in_planes):
        super(InceptionD, self).__init__()

        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_planes, 192, 1),
            BasicConv2d(192, 320, 3, stride=2)
        )
        self.branch7x7 = nn.Sequential(
            BasicConv2d(in_planes, 192, 1),
            BasicConv2d(192, 192, (1, 7), padding=(0, 3)),
            BasicConv2d(192, 192, (7, 1), padding=(3, 0)),
            BasicConv2d(192, 192, 3, stride=2)
        )
        self.branchpool = nn.AvgPool2d(3, 2)

    def forward(self, x):
        y1 = self.branch3x3(x)
        y2 = self.branch7x7(x)
        y3 = self.branchpool(x)
        return torch.cat([y1, y2, y3], 1)


class InceptionE(nn.Module):
    def __init__(self, in_planes):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_planes, 320, 1)

        self.branch3x3_1 = BasicConv2d(in_planes, 384, 1)
        self.branch3x3_2a = BasicConv2d(384, 384, (1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, (3, 1), padding=(1, 0))

        self.branch3x3stack_1 = BasicConv2d(in_planes, 448, 1)
        self.branch3x3stack_2 = BasicConv2d(448, 384, 3, padding=1)
        self.branch3x3stack_3a = BasicConv2d(384, 384, (1, 3), padding=(0, 1))
        self.branch3x3stack_3b = BasicConv2d(384, 384, (3, 1), padding=(1, 0))

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BasicConv2d(in_planes, 192, 1)
        )

    def forward(self, x):
        y1 = self.branch1x1(x)
        y2 = self.branch3x3_1(x)
        y2 = [
            self.branch3x3_2a(y2),
            self.branch3x3_2b(y2)
        ]
        y2 = torch.cat(y2, 1)
        y3 = self.branch3x3stack_1(x)
        y3 = self.branch3x3stack_2(y3)
        y3 = [
            self.branch3x3stack_3a(y3),
            self.branch3x3stack_3b(y3)
        ]
        y3 = torch.cat(y3, 1)
        y4 = self.branchpool(x)
        return torch.cat([y1, y2, y3, y4], 1)


class InceptionV3(nn.Module):
    def __init__(self, num_classes=10):
        super(InceptionV3, self).__init__()
        self.pre_layers = nn.Sequential(
            BasicConv2d(3, 32, 3, padding=1),
            BasicConv2d(32, 32, 3, padding=1),
            BasicConv2d(32, 64, 3, padding=1),
            BasicConv2d(64, 80, 1),
            BasicConv2d(80, 192, 3)
        )
        self.blockA = nn.Sequential(
            InceptionA(192, pool_planes=32),
            InceptionA(256, pool_planes=64),
            InceptionA(288, pool_planes=64)
        )
        self.blockB = InceptionB(288)
        self.blockC = nn.Sequential(
            InceptionC(768, c7x7=128),
            InceptionC(768, c7x7=160),
            InceptionC(768, c7x7=160),
            InceptionC(768, c7x7=192)
        )
        self.blockD = InceptionD(768)
        self.blockE = nn.Sequential(
            InceptionE(1280),
            InceptionE(2048)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.blockA(x)
        x = self.blockB(x)
        x = self.blockC(x)
        x = self.blockD(x)
        x = self.blockE(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def inception_v3(num_classes):
    return InceptionV3(num_classes)
