import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaConv(nn.Module):
    ''' Lambda convolution layer
    :param in_channels: input channel size
    :param out_channels: output channel size
    :param heads: multi-head query
    :param k: key depth
    :param u: intra-depth
    :param m: number of context position 
    '''
    def __init__(self, in_channels, out_channels, heads=4, k=16, u=1, m=23):
        super(LambdaConv, self).__init__()
        self.kk, self.uu, self.vv, self.mm, self.heads = k, u, out_channels // heads, m, heads
        if m > 0:
            self.local_context = True
        else:
            self.local_context = False
        self.padding = (m - 1) / 2
        self.queries = nn.Sequential(
            nn.Conv2d(in_channels, k * heads, kernel_size=1, bias=False),
            nn.BatchNorm2d(k * heads) # applying batch normalization after computing the queries is helpful
        )
        self.keys = nn.Sequential(
            nn.Conv2d(in_channels, k * u, kernel_size=1, bias=False)
        )
        self.values = nn.Sequential(
            nn.Conv2d(in_channels, self.vv * u, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.vv * u) # applying batch normalization after computing the values is helpful
        )
        self.softmax = nn.Softmax(dim=1)
        if self.local_context:
            self.embedding = nn.init.kaiming.normal_(torch.empty(self.kk, self.uu, 1, m, m), model='fan_out', nonlinearity='relu')
        else:
            self.embedding = nn.init.kaiming.normal_(torch.empty(self.kk, self.uu), mode='fan_out', nonlinearity='relu')

    def forward(self,x):
        n_batch, C, w, h = x.size()
        queries = self.queries(x).view(n_batch, self.heads, self.kk, w * h)
        softmax = self.softmax(self.keys(x).view(n_batch, self.kk, self.uu, w * h))
        values = self.values(x).view(n_batch, self.vv, self.uu, w * h)
        lambda_c = torch.einsum('bkum,bvum->bkv', softmax, values) # content lambda = key dot values 
        y_c = torch.einsum('bhkn,bkv->bhvn', queries, lambda_c) # applyig content lambda to queries
        if self.local_context:
            values = values.view(n_batch, self.uu, -1, w, h)
            lambda_p = F.conv3d(values, self.embedding, padding=(0, self.padding, self.padding))
            lambda_p = lambda_p.view(n_batch, self.kk, self.vv, w * h)
            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)
        else:
            lambda_p = torch.einsum('ku,bvun->bkvn', self.embedding, values) # lambda_p = positional embedding dot values
            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p) # applying position lambda to queries
        out = y_c + y_p
        out = out.contiguous().view(n_batch, -1, w, h)
        return out


class LambdaBottleneck(nn.Module):
    '''Lambda bottleneck layer
    :param in_planes: input channel size
    :param planes: output channel size
    :param stride: stride size
    '''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(LambdaBottleneck,self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.ModuleList([LambdaConv(planes, planes)])
        if stride != 1 or in_planes != self.expansion * planes:
            self.conv2.append(nn.AvgPool2d(kernel_size=(3,3), stride=stride, padding=(1,1)))
        self.conv2.append(nn.BatchNorm2d(planes))
        self.conv2.append(nn.ReLU())
        self.conv2 = nn.Sequential(*self.conv2)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    '''ResNet
    :param block: Bottleneck layer
    :param num_blocks: number repetitive bottleneck layer
    :param num_classes: output class size
    '''
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet,self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512*block.expansion, num_classes)
        )
    
    def _make_layer(self, block, planes, num_blocks, stride=1):
        ''' make convolution block
        :param block: bottleneck layer
        :param planes: output channel size
        :param stride: stride size
        :return covolution block
        '''
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def LambdaResNet50(num_classes):
    return ResNet(LambdaBottleneck, [3, 4, 6, 3], num_classes)


def LambdaResNet101(num_classes):
    return ResNet(LambdaBottleneck,[3, 4, 23, 3], num_classes)


def LambdaResNet152(num_classes):
    return ResNet(LambdaBottleneck, [3, 8, 36, 3], num_classes)


def LambdaResNet200(num_classes):
    return ResNet(LambdaBottleneck, [3, 24, 36, 3], num_classes)


def LambdaResNet270(num_classes):
    return ResNet(LambdaBottleneck, [4, 29, 53, 4], num_classes)


def LambdaResNet350(num_classes):
    return ResNet(LambdaBottleneck, [4, 36, 72, 4], num_classes)


def LambdaResNet420(num_classes):
    return ResNet(LambdaBottleneck, [4, 44, 87, 4], num_classes)
