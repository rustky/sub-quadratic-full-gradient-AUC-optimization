import numpy as np

def squared_hinge_loss(predictions,labels,margin):
    running_loss = 0
    I_pos = np.where(labels == 1)[0]
    I_neg = np.where(labels == -1)[0]
    for i in I_pos:
        for j in I_neg:
            z_coeff = predictions[i] - predictions[j]
            loss_clipped = margin - z_coeff
            if loss_clipped > 0:
                running_loss += loss_clipped**2
    return running_loss

