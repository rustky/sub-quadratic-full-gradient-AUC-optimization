import numpy as np

def all_pairs_square_loss(predictions,labels,margin):
    a_coeff, b_coeff, c_coeff, running_loss = 0,0,0,0
    I_pos = np.where(labels == 1)[0]
    I_neg = np.where(labels == 0)[0]
    for i in I_pos:
        z_coeff = margin - predictions[i]
        a_coeff += 1
        b_coeff += (-2)*z_coeff
        c_coeff += z_coeff**2
    for j in I_neg[0]:
        predicted_value = predictions[j]
        running_loss += a_coeff*(predicted_value**2) + b_coeff*predicted_value + c_coeff
    return running_loss
