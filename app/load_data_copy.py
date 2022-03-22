from builtins import getattr, int, type
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
import libauc.datasets
from libauc.datasets import ImbalanceGenerator
import ast


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
    root = "/projects/genomic-ml/sub-quadratic-full-gradient-AUC-optimization/data"
    dataset_str = "libauc.datasets." + dataset + "()"
    dataset_tuple = eval(dataset_str)
    first_split = ["train", "test"]
    data_split = ['features','labels']
    sets_dict = {}
    for idx,sets in enumerate(first_split):
        sets_dict[sets] = dict((key,content) 
                                for key,content in zip(data_split,dataset_tuple[idx]))
    subtrain_features, validation_features, subtrain_labels, validation_labels = \
        train_test_split(sets_dict['train']['features'], sets_dict['train']['labels'],
                        test_size = 0.2, random_state=SEED)
    sets_dict.pop('train')
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
    imratio_is_balanced = imratio
    for sets in sets_dict:
        print(sets)
        if sets == 'test':
            imratio_is_balanced = 0.5
            shuffle = False
            drop_last = True
        images, image_labels = ImbalanceGenerator(sets_dict[sets]['features'],
                                                           sets_dict[sets]['labels'],
                                                           imratio = imratio_is_balanced,
                                                           shuffle=shuffle,
                                                           random_seed=SEED)
        if(sets == 'subtrain'):
            image_label_ints = image_labels.astype(int)[:,0]
            label_counts = np.bincount(image_label_ints)
            print(label_counts)
        image_set_dict[sets] = ImageDataset(images, image_labels)


        if batch_size_str == 'full':
            batch_size_dict[sets] = len(image_set_dict[sets])
        else:
            batch_size_dict[sets] = int(batch_size_str)

        loader_dict[sets] = torch.utils.data.DataLoader(image_set_dict[sets], batch_size=batch_size_dict[sets],
                                              shuffle=shuffle, num_workers=1, pin_memory=True, drop_last=drop_last)
        shuffle = True
        drop_last = False
        imratio_is_balanced = imratio
    return loader_dict,label_counts

