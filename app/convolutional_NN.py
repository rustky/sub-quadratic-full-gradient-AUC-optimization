import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from libauc.models import ResNet20
from sklearn.metrics import roc_auc_score
from functional_square_loss import functional_square_loss
import numpy as np


def train_classifier(trainloader, testloader, loss_function, num_epochs):
    train_results = []
    imratio = 0.1
    lr = 0.1
    gamma = 500
    weight_decay = 1e-4
    margin = 1.0
    model = ResNet20(pretrained=False, last_activation='sigmoid', num_classes=1)
    if torch.cuda.is_available():
        model = model.cuda()
    AUCM = AUCMLoss(imratio=imratio)
    Proposed = loss_function()
    loss_list = [Proposed, AUCM]
    for loss_algorithm in loss_list:
        print(str(loss_algorithm))
        optimizer = PESG(model,
                         a=loss_algorithm.a,
                         b=loss_algorithm.b,
                         alpha=loss_algorithm.alpha,
                         imratio=imratio,
                         lr=lr,
                         gamma=gamma,
                         margin=margin,
                         weight_decay=weight_decay)
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            train_pred = []
            train_true = []
            model.train()
            for data, targets in trainloader:
                # get the inputs; data is a list of [inputs, labels]
                # forward + backward + optimize
                outputs = model(data)
                if loss_algorithm == AUCM:
                    loss = loss_algorithm(outputs, targets)
                else:
                    loss = loss_algorithm(outputs, targets)
                    print('ur dumb')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                train_pred.append(outputs.detach().numpy())
                train_true.append(targets.detach().numpy())

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
        test_pred.append(y_pred.numpy())
        test_true.append(test_targets.numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    val_auc = roc_auc_score(test_true, test_pred)
    model.train()
    return val_auc

