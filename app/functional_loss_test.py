import torch

def square_hinge_test(predictions, labels, margin=1):
    torch.autograd.set_detect_anomaly(True)
    labels_length = len(labels)
    augmented_predictions = torch.where(labels == 1, predictions,
                                        predictions + margin)
    augmented_indices_sorted = torch.argsort(augmented_predictions)
    predicted_value = predictions[augmented_indices_sorted]
    labels_sorted = labels[augmented_indices_sorted]
    I_pos = torch.where(labels_sorted == 1)[0]
    I_neg = torch.where(labels_sorted == 0)[0]
    N = len(I_pos)*len(I_neg)
    z_coeff = margin - predicted_value
    a_coeff = torch.cumsum((labels_sorted)/N, dim = 0)
    b_coeff = torch.cumsum((2*z_coeff*labels_sorted)/N, dim = 0)
    c_coeff = torch.cumsum(((z_coeff**2)*labels_sorted)/N, dim = 0)
    loss_values = a_coeff*(predicted_value**2) + b_coeff*predicted_value + c_coeff
    return sum(loss_values[I_neg])

def square_test(predictions,labels, margin=1):
    I_pos = torch.where(labels == 1)[0]
    I_neg = torch.where(labels == 0)[0]
    z_coeff = margin - predictions[I_pos]
    a_coeff = torch.sum(labels[I_pos])
    b_coeff = torch.sum(2*z_coeff*labels[I_pos], dim = 0)
    c_coeff = torch.sum((z_coeff**2)*labels[I_pos], dim = 0)
    predicted_value = predictions[I_neg]
    loss_values = a_coeff*(predicted_value**2) + b_coeff*predicted_value + c_coeff
    running_loss = torch.sum(loss_values)/(len(I_pos)*len(I_neg))
    return running_loss
