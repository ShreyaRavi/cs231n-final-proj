
import numpy as np
import torch
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, transform=None, pre_transform=None, post_transform=None):
        self.ds = datasets.ImageFolder(datapath)
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.ds[index]

        x = self.pre_transform(x)

        if self.transform == "Brightness":
          x = TF.adjust_brightness(x, np.random.uniform(low=0.5,high=1.5))
          x = TF.adjust_saturation(x, np.random.uniform(low=0.5,high=1.5))
          x = TF.adjust_contrast(x, np.random.uniform(low=0.5,high=1.5))
          x = TF.adjust_hue(x, np.random.uniform(low=-0.5,high=0.5))
        elif self.transform == "Rotation":
          x = TF.rotate(x, 45*np.random.uniform(low=-1.0,high=1.0))
        
        x = self.post_transform(x)
        
        return x, y

    def __len__(self):
        return len(self.ds)