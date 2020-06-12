import argparse
from pathlib import Path


ROOTPATH="/mnt/md0/new-home/joycenerd/AICUP_MangoClassification"

parser=argparse.ArgumentParser()
parser.add_argument('--data_root',type=str,default=Path(ROOTPATH).joinpath('C1-P1_Train Dev_fixed'),help='Your dataset root directory')
parser.add_argument('--model',type=str,default="efficientnet-b4",help="which model: [resenest50, resnest101, resnest200, resnest269, efficientnet-b4--]")
parser.add_argument('--lr',type=float,default=0.01,help="learning rate")
parser.add_argument('--cuda_devices',type=int,default=0,help='gpu device')
parser.add_argument('--epochs',type=int,default=300,help='num of epoch')
parser.add_argument('--num_classes',type=int,default=3,help='The number of classes for your classification problem')
parser.add_argument('--train_batch_size',type=int,default=8,help='The batch size for training data')
parser.add_argument('--num_workers',type=int,default=4,help='The number of worker while training')
parser.add_argument('--dev_batch_size',type=int,default=4,help='The batch size for development data')
parser.add_argument('--checkpoint_dir',type=str,default=Path(ROOTPATH).joinpath('checkpoint'),help='Directory to save all your checkpoint.pth')
parser.add_argument('--weight_path',type=str,help='The path of checkpoint.pth to retrieve weight')
parser.add_argument('--img_size', type=int, default=380, help='Input image size')
parser.add_argument('--test_root', type=str, default=Path(ROOTPATH).joinpath('C1-P1_Test'), help='testset path')
opt=parser.parse_args()
