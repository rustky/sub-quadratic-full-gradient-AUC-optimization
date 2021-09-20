import pdb
import torchvision
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
from libauc.datasets import CIFAR10
from libauc.datasets import ImbalanceGenerator


class ImageDataset(Dataset):
    def __init__(self, images, targets, image_size=32, crop_size=30, mode='train'):
       self.images = images.astype(np.uint8)
       self.targets = targets
       self.mode = mode
       self.transform_train = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.RandomCrop((crop_size, crop_size), padding=None),
                              transforms.RandomHorizontalFlip(),
                              transforms.Resize((image_size, image_size)),
                              ])
       self.transform_test = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Resize((image_size, image_size)),
                              ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        image = Image.fromarray(image.astype('uint8'))
        if self.mode == 'train':
            image = self.transform_train(image)
        else:
            image = self.transform_test(image)
        return image, target


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(SEED, BATCH_SIZE, imratio):
    # TODO: stratify labels in unbalanced dataset

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    pos_labels = [0, 1, 2, 3, 4]
    set_list = [trainset, testset]
    for set in set_list:
        set.class_to_idx = {'positive': 1, 'negative': -1}
        set.classes = [1, -1]
        for label_idx in range(0, len(set.targets)):
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

    return trainloader, testloader

