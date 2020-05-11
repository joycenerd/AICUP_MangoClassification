from torch.utils.data import Dataset,DataLoader
from pathlib import Path
import numpy as np
from options import opt
from PIL import Image
from torchvision import transforms
import torch
import collections
import numbers
import random
import torch.nn as nn


label_dict={
    'A':0,
    'B':1,
    'C':2
}


class MangoDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir=Path(root_dir)
        self.x=[]
        self.y=[]
        self.transform=transform
        self.num_classes=opt.num_classes

        if root_dir.name=='C1-P1_Train':
            labels=np.genfromtxt(Path(opt.data_root).joinpath('train.csv'),dtype=np.str,delimiter=',',skip_header=1)
        
        else:
            labels=np.genfromtxt(Path(opt.data_root).joinpath('dev.csv'),dtype=np.str,delimiter=',',skip_header=1)

        for label in labels:
                self.x.append(label[0])
                self.y.append(label_dict[label[1]])

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        image_path=Path(self.root_dir).joinpath(self.x[index])
        image=Image.open(image_path).convert('RGB')
        image=image.copy()

        if self.transform:
            image=self.transform(image)
        
        return image,self.y[index]


def Dataloader(dataset,batch_size,shuffle,num_workers):
    data_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    return data_loader

def _random_colour_space(x):
    output = x.convert("HSV")
    return output


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(opt.cuda_devices)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 

def make_dataset(_dir):

    colour_transform = transforms.Lambda(lambda x: _random_colour_space(x))
    # random_gaussian_noise=transforms.Lambda(RandomGaussianNoise(mean=0.5,std=0.01))

    data_transform_1=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])

    data_transform_2=transforms.Compose([
        transforms.RandomResizedCrop(224),   
        transforms.RandomChoice([
            transforms.RandomRotation((-45,45)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=10,shear=50),
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])

    data_transform_3=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomChoice([
            transforms.RandomApply([colour_transform]),
            transforms.Lambda(GaussianNoise()),
            transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0),
            transforms.RandomGrayscale(p=0.1)
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

    data_transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])

    data_set_1=MangoDataset(Path(opt.data_root).joinpath(_dir),data_transform_1)
    # print("\nAfter first\n")
    data_set_2=MangoDataset(Path(opt.data_root).joinpath(_dir),data_transform_2)
    # print("\nAfter second\n")
    data_set_3=MangoDataset(Path(opt.data_root).joinpath(_dir),data_transform_3)
    # print("\nAfter third\n")
    data_set=data_set_1+data_set_2+data_set_3

    return data_set



