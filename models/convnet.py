from __future__ import print_function

import torch
import torch.nn as nn


class ConvNet(nn.Module):

    def __init__(self, out_channel=[64,64, 64, 64], num_classes=-1):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, out_channel[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel[0]),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channel[0], out_channel[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel[1]),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(out_channel[1], out_channel[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel[2]),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(out_channel[2], out_channel[3], kernel_size=3, padding=1),
            # nn.BatchNorm2d(64, momentum=1, affine=True, track_running_stats=False),
            nn.BatchNorm2d(out_channel[3]),
            nn.ReLU())

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.outshape = out_channel[3]

        self.num_classes = num_classes
        if self.num_classes > 0:
            self.classifier = nn.Linear(self.outshape, self.num_classes)
            self.rot_classifier = nn.Linear(self.num_classes, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, is_feat=False, rot=False):
        out = self.layer1(x)
        f0 = out
        f0 = self.avgpool(f0)
        out = self.layer2(out)
        f1 = out
        f1 = self.avgpool(f1)
        out = self.layer3(out)
        f2 = out
        f2=self.avgpool(f2)
        out = self.layer4(out)
        f3 = out
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        feat = out  #out channel=64

        if self.num_classes > 0:
            out = self.classifier(out)

        if (rot):
            xy = self.rot_classifier(out)
            return [f0, f1, f2, f3, feat], (out, xy)

        if is_feat:
            return [f0, f1, f2, f3, feat], out
        else:
            return out


def convnet4(**kwargs):
    """Four layer ConvNet
    """
    model = ConvNet(**kwargs)
    return model


if __name__ == '__main__':
    model = convnet4(num_classes=64)
    data = torch.randn(2, 3, 84, 84)
    feat, logit = model(data, is_feat=True)
    print(feat[0].shape,feat[1].shape,feat[2].shape,feat[3].shape,feat[4].shape)
    print(logit.shape)
