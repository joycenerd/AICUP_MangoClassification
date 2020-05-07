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

        if self.transform:
            image=self.transform(image)
        
        return image,self.y[index]


def Dataloader(dataset,batch_size,shuffle,num_workers):
    data_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    return data_loader

def _random_colour_space(x):
    output = x.convert("HSV")
    return output

def gaussian_noise(img: np.ndarray, mean, std):
    imgtype = img.dtype
    gauss = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = np.clip((1 + gauss) * img.astype(np.float32), 0, 255)
    return noisy.astype(imgtype)


class RandomGaussianNoise(object):
    """Applying gaussian noise on the given CV Image randomly with a given probability.
        Args:
            p (float): probability of the image being noised. Default value is 0.5
        """

    def __init__(self, p=0.5, mean=0, std=0.1):
        assert isinstance(mean, numbers.Number) and mean >= 0, 'mean should be a positive value'
        assert isinstance(std, numbers.Number) and std >= 0, 'std should be a positive value'
        assert isinstance(p, numbers.Number) and p >= 0, 'p should be a positive value'
        self.p = p
        self.mean = mean
        self.std = std

    @staticmethod
    def get_params(mean, std):
        """Get parameters for gaussian noise
        Returns:
            sequence: params to be passed to the affine transformation
        """
        mean = random.uniform(-mean, mean)
        std = random.uniform(-std, std)

        return mean, std

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Image to be noised.
        Returns:
            np.ndarray: Randomly noised image.
        """
        img=np.asanyarray(img)
        if random.random() < self.p:
            return gaussian_noise(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

def make_dataset():

    colour_transform = transforms.Lambda(lambda x: _random_colour_space(x))

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
            RandomGaussianNoise(mean=0.5,std=0.01),
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

    if opt.mode=='train':
        data_set_1=MangoDataset(Path(opt.data_root).joinpath('C1-P1_Train'),data_transform_1)
        data_set_2=MangoDataset(Path(opt.data_root).joinpath('C1-P1_Train'),data_transform_2)
        data_set_3=MangoDataset(Path(opt.data_root).joinpath('C1-P1_Train'),data_transform_3)
        data_set=data_set_1+data_set_2+data_set_3

    elif opt.mode=='evaluate':
        data_set=MangoDataset(Path(opt.data_root).joinpath('C1-P1_Dev'),data_transform)

    return data_set



