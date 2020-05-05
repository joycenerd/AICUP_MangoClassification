from dataset import MangoDataset,Dataloader
from options import opt
from pathlib import Path
from torchvision import transforms
from torch.autograd import Variable
from model import RESNET


def main():
    data_transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

    train_set=MangoDataset(Path(opt.data_root).joinpath('C1-P1_Train'),data_transform)
    data_loader=Dataloader(dataset=train_set,batch_size=opt.train_batch_size,shuffle=True,num_workers=opt.train_num_workers)
    
    model=RESNET(num_classes=opt.num_classes)
    model=model.cuda(opt.cuda_devices)
    model.train()

    for epoch in range(opt.epochs):
        print(f'Epoch: {epoch+1}/{opt.epochs}')
        print('-'*len(f'Epoch: {epoch+1}/{opt.epochs}'))

        for i, (inputs,labels) in enumerate(data_loader):
            inputs=Variable(inputs.cuda(opt.cuda_devices))
            labels=Variable(labels.cuda(opt.cuda_devices))

            outputs=model(inputs)
            print(outputs.shape)
            break








if __name__=="__main__":
    main()