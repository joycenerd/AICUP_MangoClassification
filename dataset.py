from torch.utils.data import Dataset,DataLoader
from pathlib import Path
import numpy as np
from options import opt
from PIL import Image


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

