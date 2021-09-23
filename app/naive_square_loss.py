import numpy as np
import torch
import pdb


def naive_square_loss(predictions, labels, margin):
    running_loss = 0
    I_pos = torch.where(labels == 1)[0]
    I_neg = torch.where(labels == 0)[0]
    for j in I_pos:
        for k in I_neg:
            z_coeff = predictions[j] - predictions[k]
            running_loss += (margin - z_coeff)**2
    return running_loss
