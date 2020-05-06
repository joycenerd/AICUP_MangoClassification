from torch.utils.data import Dataset,DataLoader
from pathlib import Path
import numpy as np
from options import opt
from PIL import Image
from torchvision import transforms


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
            for label in labels:
                self.x.append(label[0])
                self.y.append(label_dict[label[1]])
        else:
            labels=np.genfromtxt(Path(opt.data_root).joinpath('dev.csv'),dtype=np.str,delimiter=',',skip_header=1)

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

def make_dataset():

    data_transform_1=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])

    data_transform_2=transforms.Compose([
        transforms.CenterCrop(224),
        transforms.RandomRotation(degrees=25),
        transforms.RandomAffine(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])

    data_transform_3=transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])

    data_transform=transforms.Compose([
        transforms.Resize(224),
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



