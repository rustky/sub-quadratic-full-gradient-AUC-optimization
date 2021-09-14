import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
import numpy as np
from functional_square_loss import functional_square_loss
from naive_square_loss import square_loss


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


def train_classifier(trainloader, loss_function):
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.0000000001, momentum=0.9)
    start = time.time()
    #Train the network
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels, 1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # print(running_loss)
    end = time.time()
    # print(end - start)
    print('Finished Training')
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)


def test_classifier(testloader):
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    classes = [1, -1]

    # # print images
    # imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % labels[j].item() for j in range(4)))

    #load saved model
    PATH = './cifar_net.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH))
    outputs = net(images)
    _, predicted = 
    print(outputs)
    print('Predicted ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
