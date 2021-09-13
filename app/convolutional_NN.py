import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
import numpy as np
import pdb
from squared_hinge_loss import squared_hinge_loss
from all_pairs_square_loss import all_pairs_square_loss
from all_pairs_squared_hinge_loss import all_pairs_squared_hinge_loss
from square_loss import square_loss


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


def train_classifier(trainloader):
    net = Net()
    start = time.time()
    #Train the network
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # forward + backward + optimize
       
            outputs = net(inputs)
            loss = square_loss(outputs, labels, 1)
            loss.backward()

            running_loss += loss.item()
            # print(running_loss)
            # if i % 20 == 19:    # print every 20 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #         (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0
            # print(i)
    end = time.time()
    print(end - start)
    print(running_loss)
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
    with torch.no_grad():
        outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j].item()]
                              for j in range(4)))
