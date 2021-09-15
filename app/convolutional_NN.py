import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
from libauc.models import ResNet20
from sklearn.metrics import roc_auc_score
import numpy as np



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_classifier(trainloader, testloader, loss_function):
    train_results = []
    lr = 0.1
    model = ResNet20(pretrained=False, last_activation='sigmoid', num_classes=1)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        train_pred = []
        train_true = []
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_function(outputs, labels, 1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_pred.append(outputs.cpu().detach().numpy())
            train_true.append(labels.cpu().detach().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_auc = roc_auc_score(train_true, train_pred)
        test_auc = test_classifier(testloader, model)
        epoch_results = dict({'loss': running_loss, 'train_auc': train_auc, 'test_auc': test_auc, 'epoch': epoch, 'lr': lr})
        train_results.append(epoch_results)
    print('Finished Training')
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

