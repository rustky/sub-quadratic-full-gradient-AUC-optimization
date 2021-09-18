import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
from libauc.losses import AUCMLoss
from libauc.optimizers import SOAP_SGD
from libauc.models import ResNet20
from sklearn.metrics import roc_auc_score
import numpy as np


def train_classifier(trainloader, testloader, loss_function):
    train_results = []
    lr = 0.1
    weight_decay = 2e-4
    model = ResNet20(pretrained=False, last_activation='sigmoid', num_classes=1)
    zero_tensor = torch.tensor([0.],requires_grad=True)
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer = SOAP_SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    Loss = AUCMLoss()
    loss_list = [Loss, loss_function]

    for loss_algorithm in loss_list:
        print(str(loss_algorithm))
        for epoch in range(10):  # loop over the dataset multiple times
            running_loss = 0.0
            train_pred = []
            train_true = []
            model.train()
            for data, targets in trainloader:
                # get the inputs; data is a list of [inputs, labels]
                # forward + backward + optimize
                outputs = model(data)
                if loss_algorithm == Loss:
                    loss = loss_algorithm(outputs, targets)
                else:
                    loss = loss_algorithm(outputs, targets, 1)
                    print('ur dumb')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                train_pred.append(outputs.cpu().detach().numpy())
                train_true.append(targets.cpu().detach().numpy())

            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            train_auc = roc_auc_score(train_true, train_pred)
            test_auc = 0
            # test_auc = test_classifier(testloader, model)
            epoch_results = dict({'loss': running_loss, 'train_auc': train_auc, 'test_auc': test_auc, 'epoch': epoch, 'lr': 0.1})
            train_results.append(epoch_results)
        print(train_results)
    return train_results


def test_classifier(testloader, model):
    model.eval()
    test_pred = []
    test_true = []
    for j, data in enumerate(testloader):
        test_data, test_targets = data
        y_pred = model(test_data)
        test_pred.append(y_pred.cpu().detach().numpy())
        test_true.append(test_targets.numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    val_auc = roc_auc_score(test_true, test_pred)
    model.train()
    return val_auc

