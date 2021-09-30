import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
from datetime import datetime
from libauc.optimizers import SGD
from libauc.models import ResNet20
from sklearn.metrics import roc_auc_score
from functional_square_loss import functional_square_loss
import numpy as np


def train_classifier(trainloader, testloader, loss_function, num_epochs, learning_rate):
    model = ResNet20(pretrained=False, last_activation='sigmoid', num_classes=1)
    if torch.cuda.is_available():
        model = model.cuda()
    results = []
    print(str(loss_function))
    optimizer = SGD(model.parameters(), lr=learning_rate)
    count = 0
    set_loaders = {"train": trainloader, "test":testloader}
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("Epoch: " + str(epoch))
        train_pred = []
        train_true = []
        model.train()
        for data, targets in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            # forward + backward + optimize
            outputs = model(data)
            loss = loss_function(outputs, targets, 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end = datetime.now()
            count = count + len(data)
        model.eval()
        epoch_res = {'epoch': epoch, 'lr': learning_rate}
        for set_name, loader in set_loaders.items():
            outputs_list = []
            targets_list = []
            for data, targets in loader:
                outputs = model(data)
                outputs_list.append(outputs.cpu().detach().numpy())
                targets_list.append(targets.cpu().detach().numpy())
            outputs_array = np.concatenate(outputs_list)
            targets_array = np.concatenate(targets_list)
            epoch_res[set_name + "_loss"] = loss_function(outputs_array, targets_array, 1)
            epoch_res[set_name + "_auc"] = roc_auc_score(outputs_array, targets_array)
        results.append(epoch_res)
    return results


def test_classifier(testloader, model):
    model.eval()
    test_pred = []
    test_true = []
    for j, data in enumerate(testloader):
        test_data, test_targets = data
        y_pred = model(test_data)
        test_pred.append(y_pred.cpu().detach().numpy())
        test_true.append(test_targets.cpu().detach().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    val_auc = roc_auc_score(test_true, test_pred)
    model.train()
    return val_auc

