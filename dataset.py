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

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    Origin: https://github.com/zhunzhong07/Random-Erasing
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

def make_dataset(_dir):

    colour_transform = transforms.Lambda(lambda x: _random_colour_space(x))

    transform = [
        transforms.RandomAffine(degrees=30,shear=50, resample=False, fillcolor=0),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3, fill=0),
        transforms.RandomRotation((-90,90), resample=False, expand=False, center=None),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.Grayscale(num_output_channels=3),
        RandomErasing(),
        RandomShift(3),
        transforms.RandomApply([colour_transform]),
    ]

    data_transform_train=transforms.Compose([
        transforms.RandomResizedCrop(opt.img_size),
        transforms.RandomApply(transform,p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])

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

