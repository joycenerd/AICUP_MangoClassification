from dataset import make_dataset,Dataloader
from options import opt
from pathlib import Path
from torchvision import transforms
from torch.autograd import Variable
from model import RESNET
import torch
import torch.nn as nn
import copy
from evaluation import eval


def train():
    train_set=make_dataset()
    data_loader=Dataloader(dataset=train_set,batch_size=opt.train_batch_size,shuffle=True,num_workers=opt.train_num_workers)
    
    model=RESNET(num_classes=opt.num_classes)
    model=model.cuda(opt.cuda_devices)
    model.train()

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(opt.epochs):
        print(f'Epoch: {epoch+1}/{opt.epochs}')
        print('-'*len(f'Epoch: {epoch+1}/{opt.epochs}'))

        training_loss = 0.0
        training_corrects = 0

        for i, (inputs,labels) in enumerate(data_loader):
            inputs=Variable(inputs.cuda(opt.cuda_devices))
            labels=Variable(labels.cuda(opt.cuda_devices))

            optimizer.zero_grad()
            outputs=model(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            training_loss += loss.item() * inputs.size(0)
            training_corrects += torch.sum(preds == labels.data)
        
        training_loss = training_loss / len(train_set)
        training_acc = float(training_corrects) / len(train_set)

        print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n')

        if training_acc > best_acc:
            best_acc = training_acc
            best_train_loss=training_loss
            best_model_params = copy.deepcopy(model.state_dict())

    print(f'Best training loss: {best_train_loss:.4f}\t Best accuracy: {best_acc:.4f}\n')
        
    model.load_state_dict(best_model_params)
    weight_path=Path(opt.checkpoint_dir).joinpath(f'model-{best_acc:.02f}-best_train_acc.pth')
    torch.save(model, str(weight_path))
        

if __name__=="__main__":
    if opt.mode=='train':
        train()
    elif opt.mode=='evaluate':
        eval()
