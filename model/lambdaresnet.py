import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaConv(nn.Module):
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
            self.embedding = nn.init.kaiming.normal_(torch.empty(self.kk, self.uu), mode='fan_out', nonlinearity='rele')

    
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
