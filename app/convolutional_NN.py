import torch
torch.backends.cudnn.benchmark = True # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
from libauc.losses import AUCMLoss # from LBFGS import LBFGS,FullBatchLBFGS
from libauc.optimizers import *
from libauc.models import ResNet20
from libauc.models import *
from LinearModel import LinearModel
from torchmetrics.classification.auroc import AUROC
import numpy as np
roc_auc_score = AUROC()
from load_data import load_data
import functional_loss_test
import sys
import os

## TODO Create tensors directly on the target device: Instead of
## calling torch.rand(size).cuda() to generate a random tensor,
## produce the output directly on the target device: torch.rand(size,
## device=torch.device('cuda')).

# https://stackoverflow.com/questions/53331247/pytorch-0-4-0-there-are-three-ways-to-create-tensors-on-cuda-device-is-there-s/53332659
dev_str = "cuda" if torch.cuda.is_available() else "cpu"
print('using device='+dev_str)
device = torch.device(dev_str)

def write_list(f, L):
    txt = "\t".join([str(x) for x in L]) + "\n"
    f.write(txt)

COLUMN_ORDER = [
    'pretrained',
    "loss_name",
    "batch_size",
    "imratio",
    "lr",
    "epoch",
    "train_our_square",
    "train_our_square_hinge",
    "train_libauc_loss",
    "train_auc",
    "test_our_square",
    "test_our_square_hinge",
    "test_libauc_loss",
    "test_auc"
]

def train_classifier(
    batch_size_str,
    imratio_str,
    loss_name,
    lr_str,
    out_dir,
    dataset,
    model,
    num_epochs=100,
    pretrained=True
):
    print(batch_size_str, imratio_str, loss_name, lr_str)
    # torch.autograd.detect_anomaly()
    batch_size = int(batch_size_str)
    imratio = float(imratio_str)
    lr = float(lr_str)
    is_interactive_job = os.getenv("SLURM_ARRAY_TASK_COUNT") is None
    use_subset = False
    print("use_subset=%s"%use_subset)
    SEED = 123
    trainloader, testloader = load_data(
        SEED, use_subset, batch_size, imratio, dataset)
    # loss_function = getattr(functional_loss_test, loss_name)
    loss_function = AUCMLoss(imratio)
    libauc_loss = AUCMLoss(imratio=imratio)
    epoch_res = {
        'pretrained':pretrained,
        "loss_name": loss_name,
        'lr': lr,
        "batch_size": batch_size,
        "imratio": imratio,
        "dataset": dataset,
        "model": model
    }
    file_key = "-".join([
        '%s=%s' % pair for pair in epoch_res.items()
    ])
    out_csv = '%s/%s.csv' % (out_dir,file_key)
    f = open(out_csv, "w")
    write_list(f, COLUMN_ORDER)
    model = eval(model + "()")
    model = model.to(device)
    # optimizer = SGD(model.parameters(), lr=lr)
    # optimizer = optim.LBFGS(model.parameters(), lr=lr)
    optimizer = PESG(model,
                    a=libauc_loss.a,
                    b=libauc_loss.b,
                    alpha=libauc_loss.alpha,
                    imratio=imratio,
                    lr=lr,
                    gamma=500,
                    margin=1,
                    weight_decay=1e-4)
    set_loaders = {
        "train": trainloader,
        "test":testloader
    }
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        epoch_res["epoch"] = epoch
        print("Epoch: " + str(epoch))
        model.train()
        for data, targets in trainloader:
            if 1 in targets:
                data = data.to(device)
                targets = targets.to(device)
                # print("targets: " + str(targets))
                outputs = model(data)
                loss = loss_function(outputs, targets)
                # optimizer.zero_grad(set_to_none=True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # def closure():
                #     optimizer.zero_grad()
                #     outputs = model(data)
                #     loss = loss_function(outputs, targets)
                #     return loss             
                # optimizer.step(closure)
        model.eval()
        with torch.no_grad():
            for set_name, loader in set_loaders.items():
                outputs_list = []
                targets_list = []
                for data, targets in loader:
                    data =  data.to(device)
                    targets = targets.to(device)
                    outputs = model(data)
                    outputs_list.append(outputs)
                    targets_list.append(targets)
                outputs_array = torch.cat(outputs_list)
                targets_array = torch.cat(targets_list).int()
                eval_dict = {"our_square":functional_loss_test.square_test, "our_square_hinge": functional_loss_test.square_hinge_test, "libauc_loss":libauc_loss, "auc":roc_auc_score}
                for eval_name, fun in eval_dict.items():
                    loss = fun(outputs_array, targets_array)
                    epoch_res[set_name + "_"+eval_name] = loss.item()
            out_list = [epoch_res[k] for k in COLUMN_ORDER]
            write_list(f, out_list)
