from .convnet import convnet4
from .resnet import resnet12
from .resnet_ssl import resnet12_ssl
from .resnet_sd import resnet12_sd
from .resnet_selfdist import multi_resnet12_kd
from .resnet import seresnet12
from .wresnet import wrn_28_10
from .MobileNetssl import MobileNet13
from .shufflenet import Shuffle1

from .resnet_new import resnet50
#from .Vit12 import vit12

model_pool = [
    'convnet4',
    'resnet12',
    'resnet12_ssl',
    'resnet12_kd',
    'resnet12_sd',
    'seresnet12',
    'wrn_28_10',
    'MobileNet13',
     'MobileNet13',
    #'vit12',

]

model_dict = {
    'MobileNet13':MobileNet13,
    'wrn_28_10': wrn_28_10,
    'convnet4': convnet4,
    'resnet12': resnet12,
    'resnet12_ssl': resnet12_ssl,
    'resnet12_kd': multi_resnet12_kd,
    'resnet12_sd': resnet12_sd,
    'seresnet12': seresnet12,
    'resnet50': resnet50,
    'MobileNet13':MobileNet13,
    #'vit12': vit12,
}
