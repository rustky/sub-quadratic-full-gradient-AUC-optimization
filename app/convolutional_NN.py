import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
from libauc.losses import AUCMLoss
from libauc.optimizers import SGD
from libauc.models import ResNet20
from sklearn.metrics import roc_auc_score
from functional_square_loss import functional_square_loss
import numpy as np


def train_classifier(trainloader, testloader, loss_function, num_epochs, learning_rate):
    model = ResNet20(pretrained=False, last_activation='sigmoid', num_classes=1)
    if torch.cuda.is_available():
        model = model.cuda()
    train_results = []
    print(str(loss_function))
    optimizer = SGD(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print(epoch)
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
        test_auc = 0
        # test_auc = test_classifier(testloader, model)
        epoch_results = dict({'loss': loss.item(), 'train_auc': train_auc, 'test_auc': test_auc, 'epoch': epoch, 'lr': learning_rate})
        train_results.append(epoch_results)
        print(epoch_results)
    return train_results


def test_classifier(testloader, model):
    model.eval()
    test_pred = []
    test_true = []
    for j, data in enumerate(testloader):
        test_data, test_targets = data
        y_pred = model(test_data)
        test_pred.append(y_pred.numpy())
        test_true.append(test_targets.numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    val_auc = roc_auc_score(test_true, test_pred)
    model.train()
    return val_auc

