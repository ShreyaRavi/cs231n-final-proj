import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from sklearn import metrics
import os

def test_model(testloader, path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # to avoid surging GPU
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
        
    model = models.resnet50(pretrained=True, progress=True)
    model.load_state_dict(torch.load(path), strict=False) # missing keys if True
    model.eval()
    model.to(device)

    preds = torch.tensor([], dtype=torch.float, device=device)
    y_true = torch.tensor([], dtype=torch.float, device=device)
    loss_calc = nn.CrossEntropyLoss()
    
    test_loss = 0
    accuracy = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device),labels.to(device)
            scores_eval = model.forward(inputs)
            batch_loss = loss_calc(scores_eval, labels)
            test_loss += batch_loss.item()

            output = F.softmax(scores_eval, dim=1)
            preds = torch.cat((preds, torch.argmax(output, dim=1).flatten().float()))
            y_true = torch.cat((y_true, labels.flatten().float()))

            equals = preds == y_true
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
    preds = preds.to('cpu').numpy()
    y_true = y_true.to('cpu').numpy()
    f1_score = metrics.f1_score(y_true, preds, average='weighted')

    print("accuracy: ", accuracy)
    print("test loss: ", test_loss)
    print("F1 score: ", f1_score)
