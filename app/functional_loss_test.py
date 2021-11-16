import torch

def square_hinge_test(predictions, labels, margin=1):
    I_pos = torch.where(labels == 1)[0]
    I_neg = torch.where(labels == 0)[0]
    torch.autograd.set_detect_anomaly(True)
    a_coeff, b_coeff, c_coeff = torch.Tensor([1]),torch.Tensor([0]),torch.Tensor([0])
    running_loss = torch.Tensor([0])
    labels_length = len(labels)
    augmented_predictions = torch.zeros(labels_length)
    for i in range(0, labels_length):
        if labels[i] == 0:
            augmented_predictions[i] = predictions[i] + margin
        else:
            augmented_predictions[i] = predictions[i]
    augmented_predictions_sorted = torch.argsort(augmented_predictions)
    predicted_value = predictions[augmented_predictions_sorted]
    labels_sorted = labels[augmented_predictions_sorted]
    z_coeff = margin - predicted_value
    a_coeff = torch.cumsum(labels_sorted, dim = 0)
    b_coeff = torch.cumsum(2*z_coeff*labels_sorted, dim = 0)
    c_coeff = torch.cumsum((z_coeff**2)*labels_sorted, dim = 0)
    loss_values = a_coeff*(predicted_value**2) + b_coeff*predicted_value + c_coeff
    return sum(loss_values[I_pos])/(len(I_pos)*len(I_neg))

def square_test(predictions,labels, margin=1):
    a_coeff, b_coeff, c_coeff, running_loss = 0,0,0,0
    I_pos = torch.where(labels == 1)[0]
    I_neg = torch.where(labels == 0)[0]
    sort = torch.argsort(labels)
    predicted_value = predictions[sort]
    sorted_labels = labels[sort]
    z_coeff = margin - predictions
    a_coeff = torch.sum(sorted_labels, dim = 0)
    b_coeff = torch.sum(2*z_coeff*sorted_labels, dim = 0)
    c_coeff = torch.sum((z_coeff**2)*sorted_labels, dim = 0)
    loss_values = a_coeff*(predictions**2) + b_coeff*predictions + c_coeff
    return sum(loss_values[I_neg])/(len(I_pos)*len(I_neg))