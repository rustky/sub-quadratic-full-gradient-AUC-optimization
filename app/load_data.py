import pdb
import torchvision
import torch
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
import numpy as np
from libauc.datasets import CIFAR10
from libauc.datasets import CAT_VS_DOG
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


class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(SEED, use_subset, batch_size, imratio, dataset):
    # TODO: stratify labels in unbalanced dataset
    (train_data, train_label), (test_data, test_label) = eval(dataset + "()")
    (train_images, train_labels) = ImbalanceGenerator(train_data, train_label, imratio=imratio, shuffle=True,
                                                      random_seed=SEED)
    (test_images, test_labels) = ImbalanceGenerator(test_data, test_label, is_balanced=True, random_seed=SEED)

    trainset = ImageDataset(train_images, train_labels)

    if use_subset:
        train_x, test_x, train_y, test_y \
            = train_test_split(train_images, train_labels, train_size=0.999, random_state=1, stratify=train_labels)
        subset = list(range(0, len(train_x), 100))
        trainset_strat = ImageDataset(train_x, train_y)
        trainset = torch.utils.data.Subset(trainset_strat, subset)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1, pin_memory=True, drop_last=True)

    testloader = torch.utils.data.DataLoader(ImageDataset(test_images, test_labels, mode='test'), batch_size=batch_size,
                                             shuffle=False, num_workers=1, pin_memory=True)

    return trainloader, testloader

