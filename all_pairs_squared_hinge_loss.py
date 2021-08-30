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
    augmented_predictions = np.argsort(augmented_predictions)
    for j in range(0,len(augmented_predictions)):
        if labels[augmented_predictions[j]] == 1:
            z_coeff = margin - predictions[augmented_predictions[j]]
            a_coeff += 1
            b_coeff += -2*z_coeff
            c_coeff += z_coeff**2
        else:
            running_loss += a_coeff*(predictions[augmented_predictions[j]]**2) + b_coeff*(predictions[augmented_predictions[j]]) + c_coeff
    return running_loss

def all_pairs_square_loss(predictions,labels,margin):
    a_coeff, b_coeff, c_coeff, running_loss = 0,0,0,0
    labels_length = len(labels)
    z_coeff = 0
    for i in range(0,labels_length):
        if(labels[i] == 1):
            z_coeff = margin - predictions[i]
            a_coeff += 1
            b_coeff += (-2)*z_coeff
            c_coeff += z_coeff**2
    for j in range(0,labels_length):
        if(labels[j] == 0):
            running_loss += a_coeff*(predictions[j]**2) + b_coeff*predictions[j] + c_coeff
    return running_loss

def full_loss(predictions,labels,margin):
    running_loss = 0
    I_pos = np.where(labels == 1)[0]
    I_neg = np.where(labels == 0)[0]
    for j in I_pos:
        for k in I_neg:
            running_loss += (margin - (predictions[j] - predictions[k])**2)
    return running_loss
# def main():
size = 100
predictions = np.random.randn(size)
labels = np.random.randint(0,2,size)
print(labels)
print(all_pairs_squared_hinge_loss(predictions, labels,1))
print(all_pairs_square_loss(predictions,labels,1))
# print(full_loss(predictions,labels,1))

