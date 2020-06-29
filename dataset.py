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
# from dataset_utils import *


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

class RandomShift(object):
    def __init__(self, shift):
        self.shift = shift
        
    @staticmethod
    def get_params(shift):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        hshift, vshift = np.random.uniform(-shift, shift, size=2)

        return hshift, vshift 
    def __call__(self, img):
        hshift, vshift = self.get_params(self.shift)
        
        return img.transform(img.size, Image.AFFINE, (1,0,hshift,0,1,vshift), resample=Image.BICUBIC, fill=1) 

def make_dataset(_dir):

    colour_transform = transforms.Lambda(lambda x: _random_colour_space(x))

    transform = [
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.Grayscale(num_output_channels=3),
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
        RandomShift(3),
        transforms.RandomApply([colour_transform]),
    ]

    data_transform_train=transforms.Compose([
        transforms.RandomResizedCrop(opt.img_size),
        transforms.RandomAffine(degrees=30,shear=50, resample=False, fillcolor=0),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3, fill=0),
        transforms.RandomRotation((-90,90), resample=False, expand=False, center=None),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply(transform,p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])

    """data_transform_2=transforms.Compose([
        transforms.RandomResizedCrop(opt.img_size ),   
        transforms.RandomChoice([
            transforms.RandomRotation((-45,45)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=10,shear=50),
            RandomShift(3)
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])

    data_transform_3=transforms.Compose([
        transforms.RandomResizedCrop(opt.img_size),
        transforms.RandomChoice([
            transforms.RandomApply([colour_transform]),
            transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0),
            transforms.RandomGrayscale(p=0.1)
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])"""

    data_transform_dev=transforms.Compose([
        transforms.Resize((opt.img_size,opt.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

    if(_dir=='C1-P1_Train'):
        data_set=MangoDataset(Path(opt.data_root).joinpath(_dir),data_transform_train)
    elif(_dir=='C1-P1_Dev'):
        data_set=MangoDataset(Path(opt.data_root).joinpath(_dir),data_transform_dev)
    
    return data_set

