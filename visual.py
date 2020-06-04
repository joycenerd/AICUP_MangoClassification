import matplotlib.pyplot as plt
from options import opt
from pathlib import Path
import numpy as np
import matplotlib as mpl


def visualization(training_loss, training_acc, dev_loss, dev_acc):
    mpl.style.use('default')
    plt.figure(figsize=[8,6])
    plt.plot(training_loss,linewidth=2.0)
    plt.plot(dev_loss,linewidth=2.0)
    plt.legend(['Training Loss', 'Dev Loss'],fontsize=18)
    plt.xlabel('Epochs',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.savefig('./figure/loss_curve.jpg')

    plt.figure(figsize=[8,6])
    plt.plot(training_acc,linewidth=2.0)
    plt.plot(dev_acc,linewidth=2.0)
    plt.legend(['Training Accuracy','Dev Accuracy'],fontsize=18)
    plt.xlabel('Epochs',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    plt.savefig("./figure/acc_curve.jpg")