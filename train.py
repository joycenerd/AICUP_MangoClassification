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
from early_stop import EarlyStopping
import adabound


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

    best_model_params_acc = copy.deepcopy(model.state_dict())
    best_model_params_loss = copy.deepcopy(model.state_dict())

    best_acc=0.0
    best_loss = float('inf')

    training_loss_list = []
    training_acc_list = []
    dev_loss_list = []
    dev_acc_list = []

    criterion = nn.CrossEntropyLoss()
    # optimizer = adabound.AdaBound(model.parameters(), lr=2e-4, final_lr=0.1)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=opt.lr, alpha=0.99, eps=1e-08, weight_decay=5e-4, momentum=0.9, centered=False)
    step = 0

    # scheduler = scheduler = StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True, cooldown=1)
    record=open('record.txt','w')

    early_stopping = EarlyStopping(patience=20, verbose=True)

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

        scheduler.step(dev_loss)
        early_stopping(dev_loss, model)
        if early_stopping.early_stop:
            print("Early Stopping")
            break

        if dev_acc > best_acc:
            best_acc = dev_acc
            best_acc_dev_loss = dev_loss

            best_train_acc=training_acc
            best_train_loss=training_loss 

            best_model_params_acc = copy.deepcopy(model.state_dict())
        
        if dev_loss < best_loss:
            the_acc = dev_acc
            best_loss = dev_loss

            the_train_acc = training_acc
            the_train_loss =  training_loss

            best_model_params_loss = copy.deepcopy(model.state_dict())

        if (epoch+1)%50==0:
            model.load_state_dict(best_model_params_loss)
            weight_path=Path(opt.checkpoint_dir).joinpath(f'model-{epoch+1}epoch-{best_loss:.02f}-loss-{the_acc:.02f}-acc.pth')
            torch.save(model,str(weight_path))

            model.load_state_dict(best_model_params_acc)
            weight_path=Path(opt.checkpoint_dir).joinpath(f'model-{epoch+1}epoch-{best_acc:.02f}-acc.pth')
            torch.save(model,str(weight_path))

            record.write(f'{epoch+1}\n')
            record.write(f'Best training loss: {best_train_loss:.4f}\tBest training accuracy: {best_train_acc:.4f}\n')
            record.write(f'Best dev loss: {best_acc_dev_loss:.4f}\tBest dev accuracy: {best_acc:.4f}\n\n')
            visualization(training_loss_list, training_acc_list, dev_loss_list, dev_acc_list, epoch+1)


        """
        if (epoch+1) == 100:
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, verbose=True, cooldown=1)
            early_stopping = EarlyStopping(patience=10, verbose=True)"""
        
        """
        if (epoch+1) >= 50:
            early_stopping(dev_loss,model)
            if early_stopping.early_stop:
                print("Early Stoppping")
                break"""

    print('Based on best accuracy:')
    print(f'Best training loss: {best_train_loss:.4f}\t Best training accuracy: {best_train_acc:.4f}')
    print(f'Best dev loss: { best_acc_dev_loss:.4f}\t Best dev accuracy: {best_acc:.4f}\n')
        
    model.load_state_dict(best_model_params_acc)
    weight_path=Path(opt.checkpoint_dir).joinpath(f'model-{best_acc:.02f}-best_acc.pth')
    torch.save(model, str(weight_path))

    print('Based on best loss:')
    print(f'Best training loss: {the_train_loss:.4f}\t Best training accuracy: {the_train_acc:.4f}')
    print(f'Best dev loss: {best_loss:.4f}\t Best dev accuracy: {the_acc:.4f}\n')
        
    model.load_state_dict(best_model_params_loss)
    weight_path=Path(opt.checkpoint_dir).joinpath(f'model-{best_loss:.02f}-best_loss-{the_acc:.02f}-acc.pth')
    torch.save(model, str(weight_path))

    visualization(training_loss_list, training_acc_list, dev_loss_list, dev_acc_list, epoch+1)
        

if __name__=="__main__":
    train()
