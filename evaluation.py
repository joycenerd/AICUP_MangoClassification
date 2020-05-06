from torchvision import transforms
from dataset import make_dataset,Dataloader
from options import opt
from pathlib import Path
import torch
import torch.nn as nn
from torch.autograd import Variable


def eval():
	eval_set=make_dataset()
	data_loader=Dataloader(dataset=eval_set,batch_size=opt.eval_batch_size,shuffle=True,num_workers=opt.eval_num_workers)

	weight_path=Path(opt.checkpoint_dir).joinpath(opt.weight_path)

	model=torch.load(weight_path)
	model=model.cuda(opt.cuda_devices)
	model.eval()

	criterion = nn.CrossEntropyLoss()

	evaluation_loss=0.0
	evaluation_corrects=0

	for i,(inputs,labels) in enumerate(data_loader):
		inputs=Variable(inputs.cuda(opt.cuda_devices))
		labels=Variable(labels.cuda(opt.cuda_devices))

		outputs=model(inputs)

		_, preds = torch.max(outputs.data, 1)
		loss = criterion(outputs, labels)

		evaluation_loss += loss.item() * inputs.size(0)
		evaluation_corrects += torch.sum(preds == labels.data)

	evaluation_loss = evaluation_loss / len(eval_set)
	evaluation_acc = float(evaluation_corrects) / len(eval_set)

	print(f'Evaluation loss: {evaluation_loss:.4f}\taccuracy: {evaluation_acc:.4f}\n')

