import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
from libauc.optimizers import SGD
from libauc.models import ResNet20
from torchmetrics.classification.auroc import AUROC
import numpy as np
roc_auc_score = AUROC()
from load_data import load_data

def write_list(f, L):
    txt = "\t".join([str(x) for x in L]) + "\n"
    f.write(txt)

COLUMN_ORDER = [
    "batch_size",
    "imratio",
    "epoch",
    "lr",
    "train_loss",
    "train_auc",
    "test_loss",
    "test_auc",
]

def train_classifier(
    batch_size,
    imratio,
    loss_function,
    num_epochs,
    learning_rate
):
    use_subset = False
    SEED = 123
    trainloader, testloader = load_data(
        SEED, use_subset, batch_size, imratio)
    epoch_res = {
        "loss": loss_function.__name__,
        'lr': learning_rate,
        "batch_size": batch_size,
        "imratio": imratio,
    }
    file_key = "-".join([
        '%s=%s' % pair for pair in epoch_res.items()
    ])
    out_csv = 'results/%s.csv' % file_key
    f = open(out_csv, "w")
    write_list(f, COLUMN_ORDER)
    model = ResNet20(
        pretrained=False, last_activation='sigmoid', num_classes=1)
    if torch.cuda.is_available():
        print('using cuda')
        model = model.cuda()
    optimizer = SGD(model.parameters(), lr=learning_rate)
    set_loaders = {
        "train": trainloader,
        "test":testloader
    }
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        epoch_res["epoch"] = epoch
        print("Epoch: " + str(epoch))
        model.train()
        for data, targets in trainloader:
            outputs = model(data)
            loss = loss_function(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            for set_name, loader in set_loaders.items():
                print(set_name)
                outputs_list = []
                targets_list = []
                for data, targets in loader:
                    outputs = model(data)
                    outputs_list.append(outputs)
                    targets_list.append(targets)
                outputs_array = torch.cat(outputs_list)
                targets_array = torch.cat(targets_list).int()
                print(outputs_array.size(), targets_array.size())
                epoch_res[set_name + "_loss"] = loss_function(
                    outputs_array, targets_array).item()
                epoch_res[set_name + "_auc"] = roc_auc_score(
                    outputs_array, targets_array).item()
            out_list = [epoch_res[k] for k in COLUMN_ORDER]
            write_list(f, out_list)
