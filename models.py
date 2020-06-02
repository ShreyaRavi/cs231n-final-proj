
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def resnet_50_classify(dropout_prob):
    model = models.resnet50(pretrained=True, progress=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(2048, 512),
                                nn.ReLU(),
                                nn.Dropout(dropout_prob),
                                nn.Linear(512, 10))

    return model

def basic_cnn():
	return None

