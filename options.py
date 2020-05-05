import argparse
from pathlib import Path


ROOTPATH="/home/dmplus/AICUP_MangoClassification"

parser=argparse.ArgumentParser()
parser.add_argument('--data_root',type=str,default=Path(ROOTPATH).joinpath('C1-P1_Train Dev_fixed'),help='Your dataset root directory')
parser.add_argument('--cuda_devices',type=int,default=0,help='gpu device')
parser.add_argument('--epochs',type=int,default=100,help='num of epoch')
parser.add_argument('--num_classes',type=int,default=3,help='The number of classes for your classification problem')
parser.add_argument('--train_batch_size',type=int,default=64,help='The batch size for training data')
parser.add_argument('--train_num_workers',type=int,default=4,help='The number of worker while training')
opt=parser.parse_args()