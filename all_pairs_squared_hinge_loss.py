from numpy import random
from numpy.core.fromnumeric import ptp
# import torch
import numpy as np

def all_pairs_squared_hinge_loss(predictions, labels, margin):
    a_coeff, b_coeff, c_coeff, running_loss = 0,0,0,0
    labels_length = len(labels)
    augmented_predictions = np.zeros(labels_length)
    for i in range(0,labels_length):
        if labels[i] == 0:
            augmented_predictions[i] = predictions[i] + margin
        else:
            augmented_predictions[i] = predictions[i] 
    augmented_predictions_sorted = np.argsort(augmented_predictions)
    for j in range(0,len(augmented_predictions)):
        augmented_indicies = augmented_predictions_sorted[j]
        if labels[augmented_indicies] == 1:
            prediction_indicies = predictions[augmented_indicies]
            z_coeff = margin - prediction_indicies
            a_coeff += 1
            b_coeff += -2*z_coeff
            c_coeff += z_coeff**2
        else:
            running_loss += a_coeff*(prediction_indicies**2) + b_coeff*(prediction_indicies) + c_coeff
    return running_loss

def all_pairs_square_loss(predictions,labels,margin):
    a_coeff, b_coeff, c_coeff, running_loss = 0,0,0,0
    labels_length = len(labels)
    for i in range(0,labels_length):
        if(labels[i] == 1):
            z_coeff = margin - predictions[i]
            a_coeff += 1
            b_coeff += (-2)*z_coeff
            c_coeff += z_coeff**2
    for j in range(0,labels_length):
        if(labels[j] == 0):
            prediction_indicies = predictions[j]
            running_loss += a_coeff*(prediction_indicies**2) + b_coeff*prediction_indicies + c_coeff
    return running_loss

def square_loss(predictions,labels,margin):
    running_loss = 0
    I_pos = np.where(labels == 1)[0]
    I_neg = np.where(labels == 0)[0]
    for j in I_pos:
        for k in I_neg:
            z_coeff = predictions[j] - predictions[k]
            running_loss += (margin - z_coeff)**2
    return running_loss

size = 100
x = np.array([1,-1])
predictions = 10**np.linspace(1,7,13).astype(int)
labels = np.repeat(x,size/2)
# print(all_pairs_squared_hinge_loss(predictions, labels,1))
print(all_pairs_square_loss(predictions,labels,1))
print(square_loss(predictions,labels,1))


