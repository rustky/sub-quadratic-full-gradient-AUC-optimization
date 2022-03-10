from random import shuffle
from cgi import test
import pdb
from posixpath import split
from torchvision.datasets import *
import torch
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
import numpy as np
from libauc.datasets import *
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


def load_data(SEED, use_subset, batch_size_str, imratio, dataset):
    # TODO: stratify labels in unbalanced dataset
    set_all_seeds(SEED)
    root = "/projects/genomic-ml/sub-quadratic-full-gradient-AUC-optimization/"
    download = True
    datasets_dict = {
        "CIFAR10": {
            "train": CIFAR10(root=root,train=True,download=download),
            "test": CIFAR10(root=root,train=False,download=download)
        },
        "STL10": {
            "train": STL10(root=root,split='train',download=download),
            "test": STL10(root=root,split='test',download=download)
        },
        "MNIST": {
            "train": MNIST(root=root,train=True,download=download),
            "test": MNIST(root=root,train=False,download=download)
        }
    }  
    selected_set = datasets_dict[dataset]
    first_split = ["train", "test"]
    sets_dict = {}
    for set in first_split:
        try:
            labels = selected_set[set].targets
        except AttributeError:
            labels = selected_set[set].labels
        zero_one_idx = np.where((labels == 0) | (labels == 1))
        sets_dict[set] = {
            "labels": labels[zero_one_idx],
            "features": selected_set[set].data[zero_one_idx]
        }
    subtrain_features, validation_features, subtrain_labels, validation_label = \
        train_test_split(sets_dict['train']['features'], sets_dict['train']['labels'],
                        test_size = 0.2, random_state=SEED)
    second_split = ['subtrain','validation']
    image_set_dict = {}
    batch_size_dict = {}
    loader_dict = {}
    shuffle = True
    drop_last = False
    for sets in second_split:
        sets_dict[sets] = {
            'labels': eval(sets+"_labels"),
            'features': eval(sets+"_features")
        }
        if sets == 'test':
            imratio = 0.5
            shuffle = False
            drop_last = True
        images, image_labels = ImbalanceGenerator(set_dict[sets]['features'],
                                                           sets_dict['sets']['labels'],
                                                           imratio = imratio,
                                                           shuffle=shuffle,
                                                           random_seed=SEED)
        image_set_dict[sets] = ImageDataset(images, image_labels)


        if batch_size_str == 'full':
            batch_size_dict[sets] = len(image_set_dict[sets])
        else:
            batch_size_dict[sets] = int(batch_size_str)

        loader_dict[sets] = torch.utils.data.DataLoader(image_set_dict[sets], batch_size=batch_size_dict[sets],
                                              shuffle=shuffle, num_workers=1, pin_memory=True, drop_last=drop_last)
        shuffle = True
        drop_last = False
    
    return loader_dict['subtrain'], loader_dict['validation'], loader_dict['test']

