from builtins import print
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
from load_data_copy import load_data
import functional_loss_test
import sys
import os
from time import gmtime, strftime
import pdb

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
    "subtrain_our_square",
    "subtrain_our_square_hinge",
    "subtrain_libauc_loss",
    "subtrain_log_loss",
    "subtrain_weighted_log_loss",
    "subtrain_auc",
    "validation_our_square",
    "validation_our_square_hinge",
    "validation_libauc_loss",
    "validation_log_loss",
    "validation_weighted_log_loss",
    "validation_auc",
    "test_our_square",
    "test_our_square_hinge",
    "test_libauc_loss",
    "test_log_loss",
    "test_weighted_log_loss",
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
    optimizer_str,
    seed_str,
    num_epochs=50,
    pretrained=False
):
    print(batch_size_str, imratio_str, loss_name, lr_str, dataset, model, optimizer_str, seed_str)
    # torch.autograd.detect_anomaly()
    imratio = float(imratio_str)
    lr = float(lr_str)
    is_interactive_job = os.getenv("SLURM_ARRAY_TASK_COUNT") is None
    use_subset = False
    print("use_subset=%s"%use_subset)
    SEED = int(seed_str)
    loader_dict,label_counts = load_data(
        SEED, use_subset, batch_size_str, imratio, dataset)
    print(label_counts)
    label_weights_tensor = torch.Tensor([1/label_counts[1]])
    loss_fun_dict = {
        "square_hinge_test": getattr(functional_loss_test, "square_hinge_test"),
        "square_test": getattr(functional_loss_test,"square_test"),
        "libauc_loss": AUCMLoss(imratio=imratio),
        "logistic_loss": nn.BCEWithLogitsLoss(),
        "weighted_logistic_loss": nn.BCEWithLogitsLoss(pos_weight=label_weights_tensor) #TODO: Add new weigths
    }
    try:
        loss_function = loss_fun_dict[loss_name]
    except KeyError:
        print("The loss function you entered is not currently supported. Please select another.")
        exit()

    libauc_loss = loss_fun_dict["libauc_loss"]
    epoch_res = {
        'pretrained':pretrained,
        "loss_name": loss_name,
        'lr': lr,
        "batch_size": batch_size_str,
        "imratio": imratio,
        "dataset": dataset,
        "model": model,
        "optimizer": optimizer_str,
        "seed": seed_str
    }
    file_key = "-".join([
        '%s=%s' % pair for pair in epoch_res.items()
    ])
    out_csv = '%s/%s.csv' % (out_dir,file_key)

    write_list(out_csv, COLUMN_ORDER,'w')
    model = eval(model + "()")
    model = model.to(device)
    if optimizer_str == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr)
    elif optimizer_str == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters(), lr=lr)
    elif optimizer_str == 'PESG':
        optimizer = PESG(model,
                    a=loss_function.a,
                    b=loss_function.b,
                    alpha=loss_function.alpha,
                    imratio=imratio,
                    lr=lr,
                    gamma=500,
                    margin=1,
                    weight_decay=1e-4)
    else:
        print("This optimizer is not currently supported, please select another.")
        exit()
    # set_loaders = {
    #     "train": trainloader,
    #     "test":testloader
    # }
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        epoch_res["epoch"] = epoch
        print("Epoch: " + str(epoch))
        model.train()
        for data, targets in loader_dict['subtrain']:
            if 1 in targets:
                data = data.to(device)
                targets = targets.to(device)
                if(optimizer_str == 'LBFGS'):
                    def closure():
                        optimizer.zero_grad()
                        outputs = model(data)
                        loss = loss_function(outputs, targets.float())
                        loss.backward()
                        return loss             
                    optimizer.step(closure)
                else:
                    outputs = model(data)
                    # pdb.set_trace()
                    loss = loss_function(outputs, targets.float())
                    if(optimizer_str == 'SGD'):
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                    elif(optimizer_str == 'PESG'):
                        optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                    optimizer.step()

        epoch_res['wall_time'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        model.eval()
        with torch.no_grad():
            for set_name, loader in loader_dict.items():
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
                eval_dict = {"our_square":functional_loss_test.square_test,
                             "our_square_hinge": functional_loss_test.square_hinge_test,
                             "libauc_loss":libauc_loss,
                             "log_loss":loss_fun_dict["logistic_loss"],
                             "weighted_log_loss":loss_fun_dict["weighted_logistic_loss"],
                             "auc":roc_auc_score}
                for eval_name, fun in eval_dict.items():
                    if eval_name != 'auc':
                        loss = fun(outputs_array, targets_array.float())
                    else:
                        loss = fun(outputs_array, targets_array)
                    epoch_res[set_name + "_"+eval_name] = loss.item()
            out_list = [epoch_res[k] for k in COLUMN_ORDER]
            write_list(out_csv, out_list,'a')

def optimizer_function(optimizer, optimizer_str):
    if(optimizer_str == "LBFGS"):
        def closure():
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_function(outputs, targets)
            loss.backward()
            return loss             
        optimizer.step(closure)
    else:
        outputs = model(data)
        loss = loss_function(outputs, targets)
        if(optimizer_str == 'SGD'):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
        elif(optimizer_str == 'PESG'):
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
        optimizer.step()