import torch.nn as nn
import torchvision.models as models
import torch
from options import opt

class ResNet(nn.Module):
    def __init__(self,block,layers,radix=1, groups=1,bottleneck_width=64,num_classes=1000,
    dilated=False,dilation=1,deep_stem=False,stem_width=64,avg_down=False,
    rectified_conv=False,rectify_avg=False,avd=False,avd_first=False,final_drop=0.0,
    droplock_prob=0,last_gamma=False,norm_layer=nn.BatchNorm2d):
    
    super(self,ResNet).__init__()

    self.cardinality=groups
    self.bottleneck_width=bottleneck_width
    self.inplanes=stem_width*2 if deep_stem else 64
    self.avg_down=avg_down
    self.last_gamma=last_gamma
    self.radix=radix
    self.avd=avd
    self.avd_first=avd_first
    self.rectified_conv=rectified_conv
    self.rectify_avg=rectify_avg

    if rectified_conv:
        from rfconv import RFConv2d
    else:
        conv_layer=nn.Conv2d

