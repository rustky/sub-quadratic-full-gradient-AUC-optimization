"""
Author: Zhuoning Yuan
Contact: yzhuoning@gmail.com
"""
import csv

from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from libauc.models import *
from libauc.datasets import *
from libauc.datasets import ImbalanceGenerator
from LinearModel import LinearModel

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


# paramaters
def libauc_function(
    batch_size_str,
    imratio_str,
    lr_str,
    model_str,
    dataset
):
    batch_size = int(batch_size_str)
    imratio = float(imratio_str)
    lr = float(lr_str)
    SEED = 123
    gamma = 500
    weight_decay = 1e-4
    margin = 1.0
    # dataloader
    (train_data, train_label), (test_data, test_label) = eval(dataset + "()")
    (train_images, train_labels) = ImbalanceGenerator(train_data, train_label, imratio=imratio, shuffle=True,
                                                    random_seed=SEED)
    (test_images, test_labels) = ImbalanceGenerator(test_data, test_label, is_balanced=True, random_seed=SEED)

    trainloader = torch.utils.data.DataLoader(ImageDataset(train_images, train_labels), batch_size=batch_size, shuffle=True,
                                            num_workers=1, pin_memory=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(ImageDataset(test_images, test_labels, mode='test'), batch_size=batch_size,
                                            shuffle=False, num_workers=1, pin_memory=True)

    # You need to include sigmoid activation in the last layer for any customized models!
    model = eval(model_str + "()")
    # model = model.cuda()

    Loss = AUCMLoss(imratio=imratio)
    optimizer = PESG(model,
                    a=Loss.a,
                    b=Loss.b,
                    alpha=Loss.alpha,
                    imratio=imratio,
                    lr=lr,
                    gamma=gamma,
                    margin=margin,
                    weight_decay=weight_decay)
    epoch_res = {
        'lr': lr,
        "batch_size": batch_size,
        "imratio": imratio,
        "dataset": dataset,
        "model": model
    }

    print('Start Training')
    print('-' * 30)
    for epoch in range(100):

        train_pred = []
        train_true = []
        model.train()
        for data, targets in trainloader:
            # data, targets = data.cuda(), targets.cuda()
            y_pred = model(data)
            train_loss = Loss(y_pred, targets)
            optimizer.zero_grad()
            train_loss.backward(retain_graph=True)
            optimizer.step()
            train_pred.append(y_pred.cpu().detach().numpy())
            train_true.append(targets.cpu().detach().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_auc = roc_auc_score(train_true, train_pred)

        model.eval()
        test_pred = []
        test_true = []
        for j, data in enumerate(testloader):
            test_data, test_targets = data
            # test_data = test_data.cuda()
            y_pred = model(test_data)
            test_loss = Loss(y_pred, test_targets)
            test_pred.append(y_pred.cpu().detach().numpy())
            test_true.append(test_targets.numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        val_auc = roc_auc_score(test_true, test_pred)
        model.train()
        print(epoch)
        # print results
        out_list = [epoch, train_loss.item(), train_auc, test_loss.item(), val_auc, optimizer.lr]
        with open('LibAUC Results/' + model_str + '/' + dataset + '-' + imratio_str + '-' + lr_str + '-' + batch_size_str , 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(out_list)
