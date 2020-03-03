import torch
import torch.nn as nn
import torchvision.models.inception


__all__ = ["inception_v2"]


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, k_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, k_size, bias=False, **kwargs),
            nn.BatchNorm2d(out_planes, eps=0.001),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class InceptionA(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_planes, n1x1, 1)
        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_planes, n3x3red, 1),
            BasicConv2d(n3x3red, n3x3, 3, padding=1)
        )
        self.branch5x5 = nn.Sequential(
            BasicConv2d(in_planes, n5x5red, 1),
            BasicConv2d(n5x5red, n5x5, 3, padding=1),
            BasicConv2d(n5x5, n5x5, 3, padding=1)
        )
        self.branchpool = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            BasicConv2d(in_planes, pool_planes, 1)
        )

    def forward(self, x):
        y1 = self.branch1x1(x)
        y2 = self.branch3x3(x)
        y3 = self.branch5x5(x)
        y4 = self.branchpool(x)
        return torch.cat([y1, y2, y3, y4], 1)


class InceptionB(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(InceptionB, self).__init__()
        self.branch1x1 = BasicConv2d(in_planes, n1x1, 1)
        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_planes, n3x3red, 1),
            BasicConv2d(n3x3red, n3x3red, (1, 3), padding=(0, 1)),
            BasicConv2d(n3x3red, n3x3, (3, 1), padding=(1, 0))
        )
        self.branch5x5 = nn.Sequential(
            BasicConv2d(in_planes, n5x5red, 1),
            BasicConv2d(n5x5red, n5x5red, (1, 3), padding=(0, 1)),
            BasicConv2d(n5x5red, n5x5red, (3, 1), padding=(1, 0)),
            BasicConv2d(n5x5red, n5x5red, (1, 3), padding=(0, 1)),
            BasicConv2d(n5x5red, n5x5, (3, 1), padding=(1, 0))
        )
        self.branchpool = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            BasicConv2d(in_planes, pool_planes, 1)
        )

    def forward(self, x):
        y1 = self.branch1x1(x)
        y2 = self.branch3x3(x)
        y3 = self.branch5x5(x)
        y4 = self.branchpool(x)
        return torch.cat([y1, y2, y3, y4], 1)


class InceptionC(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_planes, n1x1, 1)

        self.branch3x3_1 = BasicConv2d(in_planes, n3x3red, 1)
        self.branch3x3_2a = BasicConv2d(n3x3red, n3x3 // 2, (1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(n3x3red, n3x3 // 2, (3, 1), padding=(1, 0))

        self.branch5x5_1 = BasicConv2d(in_planes, n5x5red, 1)
        self.branch5x5_2 = BasicConv2d(n5x5red, n5x5 // 2, 3, padding=1)
        self.branch5x5_3a = BasicConv2d(n5x5 // 2, n5x5 // 2, (1, 3), padding=(0, 1))
        self.branch5x5_3b = BasicConv2d(n5x5 // 2, n5x5 // 2, (3, 1), padding=(1, 0))

        self.branchpool = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            BasicConv2d(in_planes, pool_planes, 1)
        )

    def forward(self, x):
        y1 = self.branch1x1(x)
        y2 = self.branch3x3_1(x)
        y2 = [
            self.branch3x3_2a(y2),
            self.branch3x3_2b(y2)
        ]
        y2 = torch.cat(y2, 1)
        y3 = self.branch5x5_1(x)
        y3 = self.branch5x5_2(y3)
        y3 = [
            self.branch5x5_3a(y3),
            self.branch5x5_3b(y3)
        ]
        y3 = torch.cat(y3, 1)
        y4 = self.branchpool(x)
        return torch.cat([y1, y2, y3, y4], 1)


class InceptionV2(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV2, self).__init__()
        self.pre_layers = BasicConv2d(3, 192, 3, padding=1)
        self.block1 = nn.Sequential(
            InceptionA(192, 64, 96, 128, 16, 32, 32),
            InceptionA(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2, 1)
        )
        self.block2 = nn.Sequential(
            InceptionB(480, 192, 96, 208, 16, 48, 64),
            InceptionB(512, 160, 112, 224, 24, 64, 64),
            InceptionB(512, 128, 128, 256, 24, 64, 64),
            InceptionB(512, 112, 114, 288, 32, 64, 64),
            InceptionB(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2, 1)
        )
        self.block3 = nn.Sequential(
            InceptionC(832, 256, 160, 320, 32, 128, 128),
            InceptionC(832, 384, 192, 384, 48, 128, 128)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifer = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifer(x)
        return x


def inception_v2(num_classes):
    return InceptionV2(num_classes)
