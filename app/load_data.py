import pdb

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    pos_labels = [0, 1, 2, 3, 4]
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                        download=True, transform=transform)

    set_list = [trainset, testset]
    for set in set_list:
        set.class_to_idx = {'positive': 1, 'negative': -1}
        set.classes = [1, -1]
        for label_idx in range(0,len(set.targets)):
            if set.targets[label_idx] in pos_labels:
                set.targets[label_idx] = 1
            else:
                set.targets[label_idx] = -1
    subset = list(range(0, len(trainset), 100))
    trainset_subset = torch.utils.data.Subset(trainset, subset)
    batch_size = int(len(trainset_subset))

    trainloader = torch.utils.data.DataLoader(trainset_subset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat',
    #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # #get some random training images
    dataiter = iter(trainloader)
    test = dataiter.next() #TODO: chagne so iter() loads binary labels

    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    return trainloader, testloader
