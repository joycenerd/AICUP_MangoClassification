from dataset import make_dataset,Dataloader
from options import opt
from pathlib import Path
from torchvision import transforms
from torch.autograd import Variable
from model.resnest.restnest import get_model
import torch
import torch.nn as nn
import copy
from evaluation import evaluation
# from resnest.torch import resnest50
from efficientnet_pytorch import EfficientNet
from model.model_utils import get_net
from tqdm import tqdm
from visual import visualization
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR


def train():
    train_set=make_dataset('C1-P1_Train')
    train_loader=Dataloader(dataset=train_set,batch_size=opt.train_batch_size,shuffle=True,num_workers=opt.num_workers)

    dev_set=make_dataset('C1-P1_Dev')
    dev_loader=Dataloader(dataset=dev_set,batch_size=opt.dev_batch_size,shuffle=True,num_workers=opt.num_workers)

    net=get_net(opt.model)
    if (opt.model[0:9] == 'resnest'):
        model=net(opt.num_classes)
    model = net
    model=model.cuda(opt.cuda_devices)

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc=0.0
    training_loss_list = []
    training_acc_list = []
    dev_loss_list = []
    dev_acc_list = []

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0001)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.lr, betas=(0.5, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=True)
    step = 0

    scheduler = scheduler = StepLR(optimizer, step_size=5, gamma=0.5, last_epoch=-1)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True, cooldown=1)
    record=open('record.txt','w')

    for epoch in range(opt.epochs):
        print(f'Epoch: {epoch+1}/{opt.epochs}')
        print('-'*len(f'Epoch: {epoch+1}/{opt.epochs}'))

        training_loss = 0.0
        training_corrects = 0

        model.train()

        for i, (inputs,labels) in enumerate(tqdm(train_loader)):
            inputs=Variable(inputs.cuda(opt.cuda_devices))
            labels=Variable(labels.cuda(opt.cuda_devices))

            optimizer.zero_grad()
            outputs=model(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            # scheduler.step()

            training_loss += loss.item() * inputs.size(0)
            training_corrects += torch.sum(preds == labels.data)
        
        training_loss = training_loss / len(train_set)
        training_acc = float(training_corrects) / len(train_set)

        training_loss_list.append(training_loss)
        training_acc_list.append(training_acc)

        print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}')

        model.eval()

        dev_loss=0.0
        dev_corrects=0

        for i,(inputs,labels) in enumerate(tqdm(dev_loader)):
            inputs=Variable(inputs.cuda(opt.cuda_devices))
            labels=Variable(labels.cuda(opt.cuda_devices))

            outputs=model(inputs)

            _,preds=torch.max(outputs.data,1)
            loss=criterion(outputs,labels)

            dev_loss+=loss.item()*inputs.size(0)
            dev_corrects+=torch.sum(preds==labels.data)

        dev_loss=dev_loss/len(dev_set)
        dev_acc=float(dev_corrects)/len(dev_set)

        dev_loss_list.append(dev_loss)
        dev_acc_list.append(dev_acc)

        print(f'Dev loss: {dev_loss:.4f}\taccuracy: {dev_acc:.4f}\n')

        if step < 17:
            scheduler.step()
        else:
            scheduler.step(dev_acc)
        step += 1

        if dev_acc > best_acc:
            best_acc = dev_acc
            best_dev_loss = dev_loss

            best_train_acc=training_acc
            best_train_loss=training_loss 

            best_model_params = copy.deepcopy(model.state_dict())

        if (epoch+1)%50==0:
            model.load_state_dict(best_model_params)
            weight_path=Path(opt.checkpoint_dir).joinpath(f'model-{epoch+1}epoch-{best_acc:.02f}-best_train_acc.pth')
            torch.save(model,str(weight_path))
            record.write(f'{epoch+1}\n')
            record.write(f'Best training loss: {best_train_loss:.4f}\tBest training accuracy: {best_train_acc:.4f}\n')
            record.write(f'Best dev loss: {best_dev_loss:.4f}\tBest dev accuracy: {best_acc:.4f}\n\n')
            visualization(training_loss_list, training_acc_list, dev_loss_list, dev_acc_list, epoch+1)

        if step == 17:
            optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True, cooldown=1)
            

    print(f'Best training loss: {best_train_loss:.4f}\t Best training accuracy: {best_train_acc:.4f}')
    print(f'Best dev loss: {best_dev_loss:.4f}\t Best dev accuracy: {best_acc:.4f}\n')
        
    model.load_state_dict(best_model_params)
    weight_path=Path(opt.checkpoint_dir).joinpath(f'model-{best_acc:.02f}-best_valid_acc.pth')
    torch.save(model, str(weight_path))

    visualization(training_loss_list, training_acc_list, dev_loss_list, dev_acc_list)
        

if __name__=="__main__":
    train()
