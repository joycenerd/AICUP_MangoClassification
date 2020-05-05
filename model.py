import torch.nn as nn
import torchvision.models as models
import torch
from options import opt

class Bottleneck(nn.Module):
    def __init__(self,in_features,out_features,stride=1,downsampling=False,expansion=4):
        super(Bottleneck,self).__init__()

        self.expansion=expansion
        self.downsampling=downsampling
        
        self.bottleneck=nn.Sequential(
            nn.Conv2d(in_channels=in_features,out_channels=out_features,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_features,out_channels=out_features,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_features,out_channels=out_features*self.expansion,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(out_features*self.expansion)
        )

        if self.downsampling:
            self.downsample=nn.Sequential(
                nn.Conv2d(in_channels=in_features,out_channels=out_features*self.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_features*self.expansion)
            )

        self.relu=nn.ReLU(inplace=True)

    
    def forward(self,x):
        residual=x
        if self.downsampling:
            residual=self.downsample(x)

        out=self.bottleneck(x)
        out+=residual
        out=self.relu(out)
        return out


class RESNET(nn.Module):
    def __init__(self,num_classes=1000):
        super(RESNET,self).__init__()
        self.expansion=4

        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.layer1=self.make_layer(in_features=64,out_features=64,block=3,stride=1)
        self.layer2=self.make_layer(in_features=256,out_features=128,block=4,stride=2)
        self.layer3=self.make_layer(in_features=512,out_features=256,block=6,stride=2)
        self.layer4=self.make_layer(in_features=1024,out_features=512,block=3,stride=2)

        self.avgpool=nn.AvgPool2d(7,stride=1)
        self.linear=nn.Linear(2048,num_classes)

        for param in self.modules():
            if isinstance(param,nn.Conv2d):
                nn.init.kaiming_normal_(param.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(param,nn.BatchNorm2d):
                nn.init.constant_(param.weight,1)
                nn.init.constant_(param.bias,0)


    def make_layer(self,in_features,out_features,block,stride):
        layers=[]
        
        layers.append(Bottleneck(in_features,out_features,stride,downsampling=True))

        for i in range(1,block):
            layers.append(Bottleneck(out_features*self.expansion,out_features))

        return nn.Sequential(*layers)

    def forward(self,x):
        x=self.conv1(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        x=self.avgpool(x)
        x=x.view(x.size(0),-1)
        x=self.linear(x)
        return x

