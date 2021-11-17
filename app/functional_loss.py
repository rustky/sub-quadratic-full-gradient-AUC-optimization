import torch

def square_hinge(predictions, labels, margin=1):
    I_pos = torch.where(labels == 1)[0]
    I_neg = torch.where(labels == 0)[0]
    torch.autograd.set_detect_anomaly(True)
    a_coeff, b_coeff, c_coeff = torch.Tensor([0]),torch.Tensor([0]),torch.Tensor([0])
    running_loss = torch.Tensor([0])
    labels_length = len(labels)
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
            running_loss += a_coeff*(predicted_value**2) + b_coeff*predicted_value + c_coeff
    return running_loss/(len(I_pos)*len(I_neg))

def square(predictions,labels, margin=1):
    a_coeff, b_coeff, c_coeff, running_loss = 0,0,0,0
    I_pos = torch.where(labels == 1)[0]
    I_neg = torch.where(labels == 0)[0]
    for i in I_pos:
        z_coeff = margin - predictions[i]
        a_coeff += 1
        b_coeff += (2)*z_coeff
        c_coeff += z_coeff**2
    for j in I_neg:
        predicted_value = predictions[j]
        running_loss += a_coeff*(predicted_value**2) + b_coeff*predicted_value + c_coeff
    return running_loss/(len(I_pos)*len(I_neg))
