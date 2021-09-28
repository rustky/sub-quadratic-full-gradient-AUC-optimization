import torch
from datetime import datetime
from libauc.optimizers import SGD
from libauc.models import ResNet20
from sklearn.metrics import roc_auc_score
from functional_square_loss import functional_square_loss
import numpy as np


def data_processed_time_limit(trainloader, testloader, loss_function, num_epochs, learning_rate, time_limit):
    model = ResNet20(pretrained=False, last_activation='sigmoid', num_classes=1)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = SGD(model.parameters(), lr=learning_rate)
    count = 0
    start = datetime.now()
    while (1):
        for epoch in range(num_epochs):  # loop over the dataset multiple times
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
                if((end - start).seconds > time_limit):
                    return count
    return count


