import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
class Block1(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1):
        super(Block1, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        #out = F.relu(self.bn2(self.conv2(out)))
        return out
class Conv2d_test(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2d_test, self).__init__()
        assert in_planes % groups == 0
        self.conv = nn.Conv2d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # 可以在这里自定义卷积核的参数
        weight = np.zeros((out_planes, in_planes//groups, kernel_size, kernel_size), dtype=np.float32)
        weight[2:, :, :, :] = 1
        self.conv.weight = nn.Parameter(torch.from_numpy(weight))

    def forward(self, x):
        output = self.conv(x)
        return output

class Channel_Shuffle(nn.Module):
    def __init__(self,groups):
        super(Channel_Shuffle, self).__init__()
        self.groups = groups

    def forward(self,x):
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        x = x.view(batch_size,self.groups,channels_per_group,height,width)
        x = x.transpose(1,2).contiguous()
        x = x.view(batch_size,-1,height,width)
        return x


class BLOCK(nn.Module):
    def __init__(self,inchannels,outchannels, stride,group):
        super(BLOCK, self).__init__()
        hidden_channels = outchannels//2
        self.shortcut = nn.Sequential()
        self.cat = True
        if stride == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(inchannels,hidden_channels,1,1,groups = group),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                Channel_Shuffle(group),
                nn.Conv2d(hidden_channels,hidden_channels,3,stride,1,groups=hidden_channels),
                nn.BatchNorm2d(hidden_channels),
                nn.Conv2d(hidden_channels,outchannels,1,1,groups=group),
                nn.BatchNorm2d(outchannels)
            )
            self.cat = False
        elif stride == 2:
            self.conv = nn.Sequential(
                nn.Conv2d(inchannels, hidden_channels, 1, 1, groups=group),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                Channel_Shuffle(group),
                nn.Conv2d(hidden_channels, hidden_channels, 3, stride, 1, groups=hidden_channels),
                nn.BatchNorm2d(hidden_channels),
                nn.Conv2d(hidden_channels, outchannels-inchannels, 1, 1, groups=group),
                nn.BatchNorm2d(outchannels-inchannels)
            )
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=3,stride=2,padding = 1)
            )
        self.relu = nn.ReLU(inplace=True)


    def forward(self,x):
        out = self.conv(x)
        x = self.shortcut(x)
        if self.cat:
            x = torch.cat([out,x],1)
        else:
            x = out+x
        return self.relu(x)


class Shuffle_v1(nn.Module):
    def __init__(self, classes,group = 1):
        super(Shuffle_v1, self).__init__()
        setting = {1:[3,24,144,288,576],
                   2:[3,24,200,400,800],
                   3:[3,24,240,480,960],
                   4:[3,24,272,544,1088],
                   8:[3,24,384,768,1536]}
        repeat = [3,7,3]
        channels = setting[group]

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels[0],channels[1],3,2,1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.block = BLOCK
        self.stages = nn.ModuleList([])

        for i,j in enumerate(repeat):
            self.stages.append(self.block(channels[1+i],channels[2+i],stride=2, group = group))
            for _ in range(j):
                self.stages.append(self.block(channels[2 + i], channels[2 + i], stride=1, group=group))

        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(256, 640, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(640)
        #self.conc=Conv2d_test(1536, 640, 3, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.conc1 = Block1(640, 640, stride=1)
        self.conc2 = Block1(640, 640, stride=1)
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(channels[-1],classes)
        )

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode = 'fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.zeros_(m.bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        for stage in self.stages:
            x = stage(x)

        x1=x[:,0:640,:,:]


        x2=x[:,640:1280,:,:]


        x3 = x[:, 1280:1536, :, :]

        x1 = self.conc2(x1)
        print(x1.shape)
        x2 = self.conc2(x2)
        x3=self.conv2(x3)
        x=torch.cat((x1,x2,x3),dim=2)
        print(x.shape)


        x = self.pool2(x)
        x = x.view(x.size(0),-1)
        return x

if __name__ == '__main__':
    input = torch.randn((1,3,224,224))
    m = Shuffle_v1(10,8)
    #summary(m, input_size=(3, 84, 84), batch_size=-1)
    out = m(input)
    print(out.shape)

