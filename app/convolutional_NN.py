import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pdb
from squared_hinge_loss import squared_hinge_loss


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_classifier(trainloader,trainset):
    net = Net()
    pos_class = np.array(range(0, 4))

    #Define a Loss Function and Optimizer
    criterion = nn. MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    #Train the network
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            bin_labels = torch.zeros(labels.size())
            for x in range(0, len(labels)):
                if labels[x].numpy() in pos_class:
                    bin_labels[x] = 1
                else:
                    bin_labels[x] = -1
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            sum_torch = torch.sum(outputs, 1)
            loss = criterion(outputs, bin_labels)
            loss.backward()
            optimizer.step()

            # print statistics
            # pdb.set_trace()

            running_loss += squared_hinge_loss(sum_torch, bin_labels, 1)
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)


def test_classifier(testloader,testset):
    pos_class = np.array(range(0, 4))
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    bin_labels = torch.zeros(labels.size())
    for x in range(0, len(labels)):
        if labels[x].numpy() in pos_class:
            bin_labels[x] = 1
        else:
            bin_labels[x] = -1
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # # print images
    # imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % bin_labels[j] for j in range(4)))

    #load saved model
    PATH = './cifar_net.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % predicted[j]
                              for j in range(4)))
