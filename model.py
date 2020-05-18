import torch.nn as nn
import torchvision.models as models
import torch
from options import opt
import math

class ResNet(nn.Module):
    def __init__(self,block,layers,radix=1, groups=1,bottleneck_width=64,num_classes=1000,
    dilated=False,dilation=1,deep_stem=False,stem_width=64,avg_down=False,
    rectified_conv=False,rectify_avg=False,avd=False,avd_first=False,final_drop=0.0,
    droplock_prob=0,last_gamma=False,norm_layer=nn.BatchNorm2d):

        super(self, ResNet).__init__()

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

        conv_kwargs={'average mode':rectify_avg} if rectified_conv else {}

        if deep_stem:
            self.conv1=nn.Sequential(
                conv_layer(3,stem_width,kernel_size=3,stride=2,padding=1,bias=False,**conv_kwargs),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width,stem_width,kernel_size=3,stride=1,padding=1,bias=False,**conv_kwargs),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width,stem_width*2,kernel_size=3,stride=1,padding=1,bias=False,**conv_kwargs)
            )
        else:
            self.conv1=conv_layer(3,64,kernel_size=7,stride=2,padding=3,bias=False,**conv_kwargs)

        self.bn1=norm_layer(self.inplanes)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1=self._make_layer(block,64,layers[0],norm_layer=norm_layer,is_first=False)
        self.layer2=self._make_layer(block,128,layers[1],stride=2,norm_layer=norm_layer)

        if dilated or dilation==4:
            self.layer3=self._make_layer(block,256,layers[2],stride=1,dilation=2,norm_layer=norm_layer,droplock_prob=droplock_prob)
            self.layer4=self._make_layer(block,512,layers[3],stride=1,dilation=4,norm_layer=norm_layer,droplock_prob=droplock_prob)
        elif dilation==2:
            self.layer3=self._make_layer(block,256,layers[2],stride=2,dilation=1,norm_layer=norm_layer,droplock_prob=droplock_prob)
            self.layer4=self._make_layer(block,512,layers[3],stride=1,dilation=2,norm_layer=norm_layer,droplock_prob=droplock_prob)
        else:
            self.layer3=self._make_layer(block,256,layers[2],stride=2,norm_layer=norm_layer,droplock_prob=droplock_prob)
            self.layer4=self._make_layer(block,512,layers[3],stride=2,norm_layer=norm_layer,droplock_prob=droplock_prob)

        self.avgpool=GlobalAvgPool2d()
        self.drop=nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc=nn.Linear(512*block.expansion,num_classes)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(2. / n))
            elif isinstance(m,norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
