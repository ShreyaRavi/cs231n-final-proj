# test.py to load the model and run the eval loop
# prints the evaluation statistics

import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from sklearn import metrics
from data_processing import load_split_train_test
from test_model import test_model
import argparse


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
    parser.add_argument("--modelPath",
                        default='baseline_resnet50_f1.bin',
                        type=str,
                        required=False,
                        help="Name of the model file to test")
    
    args = parser.parse_args()
    datadir = args.datadir
    localSplit = args.localSplit
    path = args.modelPath
    trainloader, testloader  = load_split_train_test(datadir, valid_size = .2, localSplit=localSplit)
    test_model(testloader, path)
