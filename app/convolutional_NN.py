import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
from libauc.losses import AUCMLoss
<<<<<<< HEAD
from libauc.optimizers import PESG
=======
from libauc.optimizers import SOAP_SGD
>>>>>>> a8358c7249e943d2ab8d1bb4b1b9f502dc7611dd
from libauc.models import ResNet20
from sklearn.metrics import roc_auc_score
from functional_square_loss import functional_square_loss
import numpy as np


<<<<<<< HEAD
def train_classifier(trainloader, testloader, loss_function, num_epochs):
=======
def train_classifier(trainloader, testloader, loss_function):
>>>>>>> a8358c7249e943d2ab8d1bb4b1b9f502dc7611dd
    train_results = []
    imratio = 0.1
    lr = 0.1
<<<<<<< HEAD
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
=======
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
>>>>>>> a8358c7249e943d2ab8d1bb4b1b9f502dc7611dd
            running_loss = 0.0
            train_pred = []
            train_true = []
            model.train()
            for data, targets in trainloader:
                # get the inputs; data is a list of [inputs, labels]
                # forward + backward + optimize
                outputs = model(data)
<<<<<<< HEAD
                if loss_algorithm == AUCM:
                    loss = loss_algorithm(outputs, targets)
                else:
                    loss = loss_algorithm(outputs, targets)
=======
                if loss_algorithm == Loss:
                    loss = loss_algorithm(outputs, targets)
                else:
                    loss = loss_algorithm(outputs, targets, 1)
>>>>>>> a8358c7249e943d2ab8d1bb4b1b9f502dc7611dd
                    print('ur dumb')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
<<<<<<< HEAD
                train_pred.append(outputs.detach().numpy())
                train_true.append(targets.detach().numpy())
=======
                train_pred.append(outputs.cpu().detach().numpy())
                train_true.append(targets.cpu().detach().numpy())
>>>>>>> a8358c7249e943d2ab8d1bb4b1b9f502dc7611dd

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

