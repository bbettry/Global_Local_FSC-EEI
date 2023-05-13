'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MobileNet']

class Block1(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1):
        super(Block1, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        #self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        #out = F.relu(self.bn2(self.conv2(out)))
        return out
class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, remove_linear=False, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        if remove_linear:
            self.linear = None
        else:
            self.linear = nn.Linear(640, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_classes = num_classes
        self.conc1 = Block1(640, 640, stride=1)
        self.conc2 = nn.Conv2d(384, 640, kernel_size=1, stride=1, padding=0, bias=False)
        if self.num_classes > 0:
            self.classifier = nn.Linear(640, self.num_classes)
            self.rot_classifier = nn.Linear(self.num_classes, 4)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, rot=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)

        out1=out
        x1 = out1[:, 0:640, :, :]


        x2 = out1[:, 640:1024, :, :]

        x1 = self.conc1(x1)
        x2 = self.conc2(x2)

        out= torch.cat((x1, x2), dim=2)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        feat = out  # out channel=64

        if self.num_classes > 0:
            out = self.classifier(out)

        if (rot):
            xy = self.rot_classifier(out)
            return [feat, feat, feat, out1, feat], (out, xy)

        if is_feat:
            return [feat, feat, feat, out1, feat], out
        else:
            return out
def MobileNet13(**kwargs):
    """Four layer ConvNet
    """
    model = MobileNet(**kwargs)
    return model


if __name__ == '__main__':
    model = MobileNet(num_classes=64)
    data = torch.randn(2, 3, 84, 84)
    feat, logit = model(data, is_feat=True)
    print(feat[-1].shape)
