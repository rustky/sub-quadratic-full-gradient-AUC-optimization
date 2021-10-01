import numpy as np
import torch
import pdb


def naive_square_hinge_loss(predictions, labels, margin):
    running_loss = 0
    I_pos = torch.where(labels == 1)[0]
    I_neg = torch.where(labels == 0)[0]
    num_pos = len(I_pos)
    num_neg = len(I_neg)
    for i in I_pos:
        for j in I_neg:
            z_coeff = predictions[i] - predictions[j]
            loss_clipped = margin - z_coeff
            if loss_clipped > 0:
                running_loss += loss_clipped**2
    return running_loss/(num_neg * num_pos)

