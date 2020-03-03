import torch.nn as nn


__all__ = ["nin"]


class NiN(nn.Module):
    def __init__(self, num_classes):
        super(NiN, self).__init__()
        self.features = nn.Sequential(
            self.mlpconv(3, 96, 7, 2, 2),
            nn.MaxPool2d(3, 2),
            self.mlpconv(96, 256, 5, 1, 2),
            nn.MaxPool2d(3, 2),
            self.mlpconv(256, 384, 3, 1, 1),
            nn.Dropout(),
            self.mlpconv(384, 10, 3, 1, 1)
        )
        self.classifier = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)

    def mlpconv(self, in_channels, out_channels, \
                kernel_size, stride, padding):
        blk = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, \
                      kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.ReLU(inplace=True)
        )
        return blk


def nin(num_classes):
    return NiN(num_classes)
