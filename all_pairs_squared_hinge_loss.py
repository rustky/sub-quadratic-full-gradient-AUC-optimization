import pdb
from numpy import random
from numpy.core.fromnumeric import ptp
# import torch
import numpy as np

def all_pairs_squared_hinge_loss(predictions, labels, margin):
    a_coeff, b_coeff, c_coeff, running_loss = 0,0,0,0
    labels_length = len(labels)
    augmented_predictions = np.zeros(labels_length)
    for i in range(0,labels_length):
        if labels[i] == -1:
            augmented_predictions[i] = predictions[i] + margin
        else:
            augmented_predictions[i] = predictions[i]
    augmented_predictions_sorted = np.argsort(augmented_predictions)
    for j in range(0,labels_length):
        augmented_indicies = augmented_predictions_sorted[j]
        predicted_value = predictions[augmented_indicies]
        if labels[augmented_indicies] == 1: 
            z_coeff = margin - predicted_value
            a_coeff += 1
            b_coeff += 2*z_coeff
            c_coeff += z_coeff**2
        else:
            running_loss += a_coeff*(predicted_value**2) + b_coeff*(predicted_value) + c_coeff
    return running_loss