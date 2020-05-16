# AICUP_MangoClassification

## Introduction

Code for AI CUP 2020 Mango Image Recognition Challenge: Grade Classification and Defective Classification

## Requirements

```
pytorch
torchvision
cudatoolkit=10.1
```

## Setup

1. Put your data (C1-P1_Train Dev_fixed) in this assignment folder **406410035_hw2_v1/**
2. Change the path in options.py: ROOTPATH=[your absolute path to this assignment folder]
3. install all the requirements if you haven't yet by: `while read requirement; do conda install --yes $requirement; done < requirement.txt`
4. execute the program: `python train.py` if you have any other specification you can see the options in **options.py** to set in the command
5. after executaion **record.txt** show the best results every 50 epochs
6. the **checkpoint.pth** will be save in the directory **checkpoint/** every 50 epochs

## Results

For now I get 77% accurary when using 200 epochs and only resnet50 as my model