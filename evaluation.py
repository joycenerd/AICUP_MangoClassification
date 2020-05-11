from dataset import MangoDataset,Dataloader,make_dataset
from options import opt
from pathlib import Path
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from PIL import Image

label_dict={
    'A':0,
    'B':1,
    'C':2
}


def evaluation():
	eval_set=MangoDataset(Path(opt.data_root).joinpath('C1-P1_Dev'))
	# data_loader=Dataloader(dataset=eval_set,batch_size=opt.eval_batch_size,shuffle=False,num_workers=opt.eval_num_workers)
	weight_path = Path(opt.checkpoint_dir).joinpath(opt.weight_path)

	model = torch.load(weight_path)
	model = model.cuda(opt.cuda_devices)
	model.eval()

	criterion = nn.CrossEntropyLoss()

	evaluation_loss = 0.0
	evaluation_corrects = 0

	for i,(image,label) in enumerate(eval_set):
		image_1, image_2, image_3 = eval_data_transform(image)
		label=np.asarray([label])
		label=torch.from_numpy(label)

		input_1 = Variable(image_1.cuda(opt.cuda_devices))
		input_2 = Variable(image_2.cuda(opt.cuda_devices))
		input_3 = Variable(image_3.cuda(opt.cuda_devices))
		label=Variable(label.cuda(opt.cuda_devices))

		output_1 = model(input_1)
		output_2 = model(input_2)
		output_3 = model(input_3)

		output = output_1+output_2+output_3

		_, preds = torch.max(output.data, 1)
		output/=3
		loss = criterion(output/3, label)

		evaluation_loss += loss.item() * input_1.size(0)
		evaluation_corrects += torch.sum(preds == label.data)

	evaluation_loss = evaluation_loss / len(eval_set)
	evaluation_acc = float(evaluation_corrects) / len(eval_set)

	print(f'Evaluation loss: {evaluation_loss:.4f}\taccuracy: {evaluation_acc:.4f}\n')

