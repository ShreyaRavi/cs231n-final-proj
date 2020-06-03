import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()

        model = models.resnet50(pretrained=True, progress=True)

        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Sequential(nn.Linear(2048, 512),
                                    nn.ReLU(),
                                    nn.Dropout(dropout_prob),
                                    nn.Linear(512, 10))
        self.resnet50 = model

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            # input: 224 x 224 x 3
            nn.Conv2d(3, 8, kernel_size=7), # output: 218 x 218 x 8
            nn.MaxPool2d(2, stride=2), # output: 109 x 109 x 8
            nn.ReLU(True), # output: 109 x 109 x 8
            nn.Conv2d(8, 10, kernel_size=5), # output: 105 x 105 x 10
            nn.MaxPool2d(2, stride=2), # output: 52 x 52 x 10
            nn.ReLU(True) # output: 52 x 52 x 10
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(52 * 52 * 10, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x) # xs: (52, 52, 10)
        xs = xs.view(-1, 52 * 52 * 10)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        print(x.size())

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        out = self.resnet50(x)
        return out
