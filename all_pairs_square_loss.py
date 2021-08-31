import numpy as np

def all_pairs_square_loss(predictions,labels,margin):
    a_coeff, b_coeff, c_coeff, running_loss = 0,0,0,0
    labels_length = len(labels)
    I_pos = np.where(labels == 1)[0]
    I_neg = np.where(labels == 0)[0]
    for i in I_pos:
        z_coeff = margin - predictions[i]
        a_coeff += 1
        b_coeff += (-2)*z_coeff
        c_coeff += z_coeff**2
    for j in I_neg:
        prediction_indicies = predictions[j]
        running_loss += a_coeff*(prediction_indicies**2) + b_coeff*prediction_indicies + c_coeff
    return running_loss
