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


def time_per_epoch(trainloader, testloader, loss_function, num_epochs, learning_rate):
    model = ResNet20(pretrained=False, last_activation='sigmoid', num_classes=1)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = SGD(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):  # loop over the dataset multiple times
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
            train_pred.append(outputs.cpu().detach().numpy())
            train_true.append(targets.cpu().detach().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_auc = roc_auc_score(train_true, train_pred)



