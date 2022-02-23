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
from time import gmtime, strftime

## TODO Create tensors directly on the target device: Instead of
## calling torch.rand(size).cuda() to generate a random tensor,
## produce the output directly on the target device: torch.rand(size,
## device=torch.device('cuda')).

# https://stackoverflow.com/questions/53331247/pytorch-0-4-0-there-are-three-ways-to-create-tensors-on-cuda-device-is-there-s/53332659
dev_str = "cuda" if torch.cuda.is_available() else "cpu"
print('using device='+dev_str)
device = torch.device(dev_str)

def write_list(out_csv, L, mode):
    f = open(out_csv, mode)
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
    "test_auc",
    "wall_time"
]

def train_classifier(
    batch_size_str,
    imratio_str,
    loss_name,
    lr_str,
    out_dir,
    dataset,
    model,
    num_epochs=2000,
    pretrained=False
):
    print(batch_size_str, imratio_str, loss_name, lr_str, dataset)
    # torch.autograd.detect_anomaly()
    imratio = float(imratio_str)
    lr = float(lr_str)
    is_interactive_job = os.getenv("SLURM_ARRAY_TASK_COUNT") is None
    use_subset = False
    print("use_subset=%s"%use_subset)
    SEED = 123
    trainloader, testloader = load_data(
        SEED, use_subset, batch_size_str, imratio, dataset)
    loss_function = getattr(functional_loss_test, loss_name)
    # loss_function = AUCMLoss(imratio=imratio)
    libauc_loss = AUCMLoss(imratio=imratio)
    epoch_res = {
        'pretrained':pretrained,
        "loss_name": loss_name,
        'lr': lr,
        "batch_size": batch_size_str,
        "imratio": imratio,
        "dataset": dataset,
        "model": model
    }
    file_key = "-".join([
        '%s=%s' % pair for pair in epoch_res.items()
    ])
    out_csv = '%s/%s.csv' % (out_dir,file_key)

    write_list(out_csv, COLUMN_ORDER,'w')
    model = eval(model + "()")
    model = model.to(device)
    # optimizer = SGD(model.parameters(), lr=lr)
    optimizer = optim.LBFGS(model.parameters(), lr=lr)
    # optimizer = PESG(model,
    #                 a=loss_function.a,
    #                 b=loss_function.b,
    #                 alpha=loss_function.alpha,
    #                 imratio=imratio,
    #                 lr=lr,
    #                 gamma=500,
    #                 margin=1,
    #                 weight_decay=1e-4)
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
                # outputs = model(data)
                # loss = loss_function(outputs, targets)
                # optimizer.zero_grad(set_to_none=True)
                # loss.backward()
                # For LIBAUC:
                # optimizer.zero_grad()
                # loss.backward(retain_graph=True)
                # optimizer.step()
                def closure():
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = loss_function(outputs, targets)
                    loss.backward()
                    return loss             
                optimizer.step(closure)
        epoch_res['wall_time'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
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
            write_list(out_csv, out_list,'a')
