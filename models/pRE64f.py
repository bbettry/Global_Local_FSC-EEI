from __future__ import print_function
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
def gen_mask(row, col, percent=0.7, num_zeros=None):
    if num_zeros is None:
        # Total number being masked is 0.5 by default.
        num_zeros = int((row * col) * percent)

    mask = np.hstack([
    	np.zeros(num_zeros),
        np.ones(row * col - num_zeros)])

    np.random.shuffle(mask)
    return mask.reshape(row, col)
class LinearFunction(torch.autograd.Function):
    """
    autograd function which masks it's weights by 'mask'.
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask

        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        ctx.save_for_backward(input, weight, bias, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask

        # if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask
class CustomizedLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True, mask=0.1):
        """
        Argumens
        ------------------
        mask [numpy.array]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flg of bias.
        """
        super(CustomizedLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute.
        self.weight = nn.Parameter(torch.Tensor(
            self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(
            	torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Initialize the above parameters (weight & bias).
        self.init_params()

        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.float).t()
            self.mask = nn.Parameter(mask, requires_grad=False)
            # print('\n[!] CustomizedLinear: \n', self.weight.data.t())
        else:
            self.register_parameter('mask', None)

    def init_params(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(
        	input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}, mask={}'.format(
            self.input_features, self.output_features,
            self.bias is not None, self.mask is not None)
class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        #self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        #self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        #out = F.relu(self.bn2(self.conv2(out)))
        return out

class ConvNet(nn.Module):

    def __init__(self, out_channel=[64, 64, 64, 64], num_classes=-1):
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
        self.preparer1=Block(64,640,stride=1)
        self.preparer2 = Block(64, 640, stride=1)
        self.preparer3 = Block(64, 640, stride=1)
        self.preparer4 = Block(64, 640, stride=1)
        self.preparer5 = Block(64, 640, stride=1)
        self.preparer6 = Block(64, 640, stride=1)
        self.preparer7 = Block(64, 640, stride=1)
        self.preparer8 = Block(64, 640, stride=1)
        self.preparer9 = Block(64, 640, stride=1)
        self.preparer10 = Block(64, 640, stride=1)
        self.outshape = out_channel[3]
        self.num_classes = num_classes
        if self.num_classes > 0:
            # self.classifier = CustomizedLinear(640, 64, bias=True, mask=None)
            self.classifier = nn.Linear(640,64)
            #self.classifier = CustomizedLinear(640,100,
                                               #mask=gen_mask(640, 100, 0.7))

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
        out = self.layer2(out)
        f1 = out
        out = self.layer3(out)
        f2 = out
        out = self.layer4(out)
        out1=self.preparer1(out)
        out2 = self.preparer2(out)
        out3 = self.preparer3(out)
        out4 = self.preparer4(out)
        out5= self.preparer5(out)
        out6 = self.preparer6(out)
        out7 = self.preparer7(out)
        out8 = self.preparer8(out)
        out9 = self.preparer9(out)
        out10 = self.preparer10(out)
        out=torch.cat((out1,out2,out3,out4,out5,out6,out7,out8,out9,out10),dim=1)
        f3=out
        out = self.avgpool(out)

        out = out.view(out.size(0), -1)



        feat = out

        #out channel=64

        if self.num_classes > 0:
            out = self.classifier(out)


        if (rot):
            xy = self.rot_classifier(out)
            return [f0,f1, f2, f3,feat], (out,xy)

        if is_feat:
            return [f0,f1, f2, f3,feat], out
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
