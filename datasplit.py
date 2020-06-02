import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from sklearn import metrics
from data_processing import load_split_train_test
from resnet50_baseline import train_model
import collections
from collections import defaultdict
import argparse
import csv
import random
from shutil import copyfile
import os

def getDrivertoImageDictionary(filepath):
    driver2imagelist = collections.defaultdict(list) 
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        count = 0
        #import pdb
        #pdb.set_trace() 
        for row in reader:
            if(count == 0):
                count+=1
                continue
            subject = row[0]
            classname = row[1]
            img = row[2]
            driver2imagelist[subject].append((img, classname))
            count +=1
    return driver2imagelist

def makeDataSet(driver2imagelist,numb_drivers):
    drivers = list(driver2imagelist.keys())
    #print(drivers)
    test_drivers_list = random.sample(range(0,len(drivers)),numb_drivers)
    test_set = []
    for d in test_drivers_list:
        driver = drivers[d]
        # add the list of images to the main list
        test_set.extend(driver2imagelist[driver])
        
    train_set = []
    for d in range(len(drivers)):
        if d in test_drivers_list:
            continue
        driver = drivers[d]
        train_set.extend(driver2imagelist[driver])
 
    return (train_set, test_set)

def writeDataSet(train_set, test_set, inputdir):
    # make the directories
    os.mkdir('imgs_out')    
    os.mkdir('imgs_out/train')
    for i in range(0,10):
        os.mkdir('imgs_out/train/c'+str(i))
        
    for (img, classname) in train_set:
        src =  inputdir+ "/train/" +classname+"/"+img
        dst = "imgs_out/train/"+classname +"/"+img
        copyfile(src, dst)
    
    os.mkdir('imgs_out/test')
    for i in range(0,10):
        os.mkdir('imgs_out/test/c'+str(i))
    for (img, classname) in test_set:
        src =  inputdir+ "/train/" +classname+"/"+img
        dst = "imgs_out/test/"+classname +"/"+img
        copyfile(src, dst)

if __name__ == "__main__":
  random.seed(64)
  #filepath = "C:/Users/19498/Downloads/CS231n/state-farm-distracted-driver-detection/driver_imgs_list.csv"
  parser = argparse.ArgumentParser()
  ## Required parameters
  parser.add_argument("--inputdir",
                  default="C:/Users/19498/Downloads/CS231n/state-farm-distracted-driver-detection/imgs",
                  type=str,
                  required=False,
                  help="The imgs directory")
  parser.add_argument("--drivercsv",
                      default="C:/Users/19498/Downloads/CS231n/state-farm-distracted-driver-detection/driver_imgs_list.csv",
                      type=str,
                      required=False,
                      help="The driver csv path")
  args = parser.parse_args()  
  filepath = args.drivercsv
  inputdir = args.inputdir
  print("Building the Driver to Image Dictionary")
  driver2imagelist = getDrivertoImageDictionary(filepath)
  print("Making the train and test splits")
  train_set, test_set = makeDataSet(driver2imagelist, 5)
  print("Write the new train and test to the imgs out files")
  writeDataSet(train_set, test_set,inputdir)