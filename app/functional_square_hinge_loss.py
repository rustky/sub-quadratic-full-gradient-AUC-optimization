import pdb
from numpy import random
from numpy.core.fromnumeric import ptp
import numpy as np
import torch

def functional_square_hinge_loss(predictions, labels, margin):
    torch.autograd.set_detect_anomaly(True)
    a_coeff, b_coeff, c_coeff, running_loss = 0,0,0,0
    labels_length = len(labels)
    I_pos = torch.where(labels == 1)[0]
    I_neg = torch.where(labels == 0)[0]
    num_pos = len(I_pos)
    num_neg = len(I_neg)
    augmented_predictions = torch.zeros(labels_length)
    for i in range(0, labels_length):
        if labels[i] == 0:
            augmented_predictions[i] = predictions[i] + margin
        else:
            augmented_predictions[i] = predictions[i]
    augmented_predictions_sorted = torch.argsort(augmented_predictions)
    for j in range(0, labels_length):
        augmented_indicies = augmented_predictions_sorted[j]
        predicted_value = predictions[augmented_indicies]
        if labels[augmented_indicies] == 1: 
            z_coeff = margin - predicted_value
            a_coeff += 1
            b_coeff += 2*z_coeff
            c_coeff += z_coeff**2
        else:
            running_loss += a_coeff*(predicted_value**2) + b_coeff*(predicted_value) + c_coeff
    return running_loss/(num_neg *  num_pos)