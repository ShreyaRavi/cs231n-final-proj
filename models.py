
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from spatial_transform import SpatialTransform

def resnet_50_classify(dropout_prob):
    model = models.resnet50(pretrained=True, progress=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(2048, 512),
                                nn.ReLU(),
                                nn.Dropout(dropout_prob),
                                nn.Linear(512, 10))

    return model


def spat_transform_resnet50():
	model = SpatialTransform()
	return model


# untested so far -- will prob take a while to train
def resnet50_less_frozen(dropout_prob, num_layers_unfreeze=2):
	model = models.resnet50(pretrained=True, progress=True)

    layer_cnt = 0
	for child in model.children():
		layer_cnt += 1
		if layer_cnt < (10 - num_layers_unfreeze):
		    for param in child.parameters():
		        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(2048, 512),
                            nn.ReLU(),
                            nn.Dropout(dropout_prob),
                            nn.Linear(512, 10))

    return model

def basic_cnn():
	return None

