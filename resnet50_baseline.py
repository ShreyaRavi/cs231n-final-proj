import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from sklearn import metrics
import os


def train_model(trainloader, testloader, model, experiment_name, epochs=1, learning_rate=0.003, progress_steps=10):
    
    SAVE_MODEL_PATH = 'baseline_resnet50_' + experiment_name + '.bin'
    print("SAVE_MODEL_PATH:",SAVE_MODEL_PATH)
    #os.mkdir("modelfiles")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loss_calc = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    model.to(device)

    model.train()


    steps = 0
    running_loss = 0
    train_losses, test_losses = [], []
    bestF1 = float('-inf')
    bestEpoch = 0
    bestP = 0
    bestR = 0
    for epoch in range(epochs):
        
        preds = torch.tensor([], dtype=torch.float, device=device)
        y_true = torch.tensor([], dtype=torch.float, device=device)

        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            scores = model.forward(inputs)
            loss = loss_calc(scores, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            #if steps % progress_steps == 0:
             #   print("Training Loss: ", loss)

        test_loss = 0
        accuracy = 0

        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device),labels.to(device)
                scores_eval = model.forward(inputs)
                batch_loss = loss_calc(scores_eval, labels)
                test_loss += batch_loss.item()

                print("input shape: ", inputs.shape)

                # scores: (N, C), softmax across classes (C)
                # output: (N, C)
                output = F.softmax(scores_eval, dim=1)
                # print("output shape: ", output.shape)

                # preds: (N,)
                preds = torch.cat((preds, torch.argmax(output, dim=1).flatten().float()))
                y_true = torch.cat((y_true, labels.flatten().float()))

                # debugging
                #print("y_true shape: ", y_true.shape)
                #print("preds shape: ", preds.shape)
                

                equals = preds == y_true
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        preds = preds.to('cpu').numpy()
        y_true = y_true.to('cpu').numpy()
        f1_score = metrics.f1_score(y_true, preds, average='weighted')
        precision_score = metrics.precision_score(y_true,preds,average='weighted')
        recall_score = metrics.recall_score(y_true,preds,average='weighted')
        if (f1_score > bestF1):
            #import pdb
            #pdb.set_trace()
            bestF1 = f1_score
            bestEpoch = epoch
            bestR = recall_score
            bestP = precision_score
            if os.path.exists(SAVE_MODEL_PATH):
                print("Removing SAVE MODEL PATH", SAVE_MODEL_PATH)
                os.remove(SAVE_MODEL_PATH)
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print("Saved model path:", SAVE_MODEL_PATH)

        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))                    
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/progress_steps:.3f}.. "
              f"Test loss: {test_loss/len(testloader):.3f}.. "
              f"Test accuracy: {accuracy/len(testloader):.3f}"
              f"F1 score: {f1_score:.3f}"
              f"Precision score: {precision_score:.3f}"
              f"F1 score: {recall_score:.3f}"
            )
        running_loss = 0
        model.train()
    print("Best F1 score: ", bestF1)
    print("Best epoch:",bestEpoch)
    print("Best P", bestP)
    print("Best R", bestR)



