import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.autograd import Variable
from options import opt


ROOTDIR="/mnt/hdd1/home/joycenerd/AICUP_MangoClassification"

label_dict = {
    0 : 'A',
    1 : 'B',
    2 : 'C'
}

def test():
    data_transform=transforms.Compose([
        transforms.Resize((opt.img_size,opt.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

    classes = opt.num_classes
    
    model_path = Path(opt.checkpoint_dir).joinpath(opt.weight_path)
    model= torch.load(str(model_path))
    model =  model.cuda(opt.cuda_devices)
    model.eval()

    sample_submission=pd.read_csv(Path(ROOTDIR).joinpath('test_example.csv'))
    submission=sample_submission.copy()

    for i,filename in enumerate(sample_submission['image_id']):
        image_path = Path(opt.test_root).joinpath(filename)
        image = Image.open(image_path).convert('RGB')
        image = data_transform(image).unsqueeze(0)
        
        inputs = Variable(image.cuda(opt.cuda_devices))
        outputs =  model(inputs)

        _,preds = torch.max(outputs.data,1)
        submission['label'][i] =  label_dict[preds[0].item()]

        print(filename + ' complete')

    submission.to_csv(Path(ROOTDIR).joinpath('submission.csv'), index=False)


if __name__=='__main__':
    test()