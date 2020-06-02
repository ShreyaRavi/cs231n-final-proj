import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from sklearn import metrics
from data_processing import load_split_train_test
from resnet50_baseline import train_model
import argparse
from models import resnet_50_classify


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        ## Required parameters
    parser.add_argument("--datadir",
                        default=None,
                        type=str,
                        required=True,
                        help="The imgs data dir")
    parser.add_argument("--localSplit",
                        default=-1,
                        type=int,
                        required=False,
                        help="Size of the smaller train split. Will also make a test split of the same size")
    parser.add_argument("--num_epochs",
                        default=1,
                        type=int,
                        required=False,
                        help="Number of Epochs. Default 1")
    parser.add_argument("--learning_rate",
                        default=0.003,
                        type=int,
                        required=False,
                        help="Learning rate. Default 0.003")
    parser.add_argument("--split_mode",
                    default=False,
                    type=bool,
                    required=False,
                    help="Set to False if you want the custom train and test to be used")
    args = parser.parse_args()
    datadir = args.datadir
    localSplit = args.localSplit
    split_mode = args.split_mode
    num_epochs = args.num_epochs
    print("num_epochs:", num_epochs)
    learning_rate = args.learning_rate
    print("learning_rate", learning_rate)
    print("Data dir:", datadir)
    trainloader, testloader  = load_split_train_test(datadir, split_mode,valid_size = .2, localSplit=localSplit)
    model = resnet_50_classify(dropout_prob = 0.2)
    train_model(trainloader, testloader, model, epochs=num_epochs, learning_rate=learning_rate, progress_steps=10)
    