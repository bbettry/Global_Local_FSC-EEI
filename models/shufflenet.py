import torch
import torch.nn as nn

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
    def __init__(self, group = 4,num_classes=64):
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
        self.num_classes = num_classes
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(channels[-1],self.num_classes)
        )

        if self.num_classes > 0:
            self.classifier =nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(channels[-1],self.num_classes)
        )
            self.rot_classifier = nn.Linear(self.num_classes, 4)

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

    def forward(self, x, is_feat=False, rot=False):
        x = self.conv1(x)
        x = self.pool1(x)
        for stage in self.stages:
            x = stage(x)
            out1 =x
        x = self.pool2(x)
        x = x.view(x.size(0),-1)
        #x = self.fc(x)
        feat=x

        if self.num_classes > 0:
            out = self.classifier(feat)

        if (rot):
            xy = self.rot_classifier(out)
            return [feat, feat, feat, out1, feat], (out, xy)

        if is_feat:
            return [feat, feat, feat, out1, feat], out
        else:
            return out
def Shuffle1(**kwargs):
    """Four layer ConvNet
    """
    model = Shuffle_v1(**kwargs)
    return model
if __name__ == '__main__':
    input = torch.empty((1,3,224,224))
    m = Shuffle_v1(10,8)
    out = m(input)
    print(out)
