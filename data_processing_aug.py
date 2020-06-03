import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import csv
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import ConcatDataset
from augment import TransformedDataset
from PIL import Image
import os


def load_split_train_test(datadir, split_mode, valid_size = .2,localSplit = -1):

    train_transforms = transforms.Compose([
                                       #transforms.Resize(224),
                                       transforms.Resize(256),
                                       transforms.RandomCrop(224),
                                       # Noise injection - augmentation
                                       #transforms.RandomRotation([+25,+45]),
                                       #transforms.RandomRotation([+315,+335]),
                                       #transforms.RandomHorizontalFlip(0.5),
                                       #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], # PyTorch recommends these but in this
                                                            [0.229, 0.224, 0.225]) # case I didn't get good results
                                       ])

    test_transforms = transforms.Compose([
                                      #transforms.Resize(224),
                                      transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                      ])
                                      
    
    train_idx = None
    test_idx = None 
    train_data = None
    test_data = None                                 
    if split_mode:
      train_data = datasets.ImageFolder(datadir+"/train", transform=train_transforms)
      
      test_data = datasets.ImageFolder(datadir+"/train", transform=test_transforms)
      
      num_train = len(train_data)
      print("Initial Number of Train Samples", num_train)
      
      #splitTrain = int(np.floor( (1 - valid_size) * num_train))
      #print("New Train Number of Samples:", splitTrain)
      
      
      indices = list(range(num_train))
      np.random.shuffle(indices)

      if(localSplit == -1):
          splitTrain = int(np.floor( (1 - valid_size) * num_train))
          split = int(np.floor(valid_size * num_train))
          train_idx, test_idx = indices[split:], indices[:split]
          print("New Train Number of Samples:", len(train_idx))
          print("New Test Number of Samples:", len(test_idx))
      else:
          split = localSplit
          train_idx, test_idx = indices[0:split], indices[split: split+split]
          print("New Train Number of Samples:", len(train_idx))
          print("New Test Number of Samples:", len(test_idx))
     
    else:
      print("Performing Data Augmentation")
      pre_transform = transforms.Compose([
                      transforms.Resize(256),
                      transforms.RandomCrop(224)
                    ])

      post_transform = transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])
                    ])
      image_transform = transforms.ToPILImage()
          
      train_data_orig = TransformedDataset(datapath=datadir+"/train",
                                            pre_transform=pre_transform, post_transform=post_transform)
      train_data_rotate = TransformedDataset(datapath=datadir+"/train", transform="Rotation",
                                            pre_transform=pre_transform, post_transform=post_transform)
      train_data_bright = TransformedDataset(datapath=datadir+"/train", transform="Brightness",
                                            pre_transform=pre_transform, post_transform=post_transform)

      print("Original Dataset Size:",len(train_data_orig))
      #os.mkdir("augmented_samples")
      #os.chdir("augmented_samples")
      #for i in range(10):
      #  x, y = train_data_orig[i]
      #  x = image_transform(x)
      #  x.save("orig_train_"+str(i)+".jpeg", "JPEG")

      #  x, y = train_data_bright[i]
      #  x = image_transform(x)
      #  x.save("bright_train_"+str(i)+".jpeg", "JPEG")

      #  x, y = train_data_rotate[i]
      #  x = image_transform(x)
      #  x.save("rotate_train_"+str(i)+".jpeg", "JPEG")

      datasets_list = [train_data_orig,train_data_bright,train_data_rotate]
      train_data = ConcatDataset(datasets_list)
      print("Augmented Dataset Size:", len(train_data))
      


      train_idx = np.arange(len(train_data))
      np.random.shuffle(train_idx)
      
      test_data = datasets.ImageFolder(datadir+"/test", transform=test_transforms)
      test_idx = np.arange(len(test_data))
      np.random.shuffle(test_idx)

    
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=64)
    
    print(trainloader)
    print(testloader)
    
    
    return trainloader, testloader
       
