import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
from libauc.optimizers import SGD
from libauc.models import ResNet20
from torchmetrics.classification.auroc import AUROC
import numpy as np
roc_auc_score = AUROC()

def train_classifier(
    trainloader,
    testloader,
    loss_function,
    num_epochs,
    learning_rate
):
    model = ResNet20(
        pretrained=False, last_activation='sigmoid', num_classes=1)
    if torch.cuda.is_available():
        print('using cuda')
        model = model.cuda()
    optimizer = SGD(model.parameters(), lr=learning_rate)
    set_loaders = {
        "train": trainloader,
        "test":testloader
    }
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("Epoch: " + str(epoch))
        model.train()
        for data, targets in trainloader:
            outputs = model(data)
            loss = loss_function(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            epoch_res = {'epoch': epoch, 'lr': learning_rate}
            for set_name, loader in set_loaders.items():
                print(set_name)
                outputs_list = []
                targets_list = []
                for data, targets in loader:
                    outputs = model(data)
                    outputs_list.append(outputs)
                    targets_list.append(targets)
                outputs_array = torch.cat(outputs_list)
                targets_array = torch.cat(targets_list).int()
                print(outputs_array.size(), targets_array.size())
                epoch_res[set_name + "_loss"] = loss_function(
                    outputs_array, targets_array).item()
                epoch_res[set_name + "_auc"] = roc_auc_score(
                    outputs_array, targets_array).item()
            print(epoch_res)
