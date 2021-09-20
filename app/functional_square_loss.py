import numpy as torch
import pdb
import torch


class functional_square_loss(torch.nn.Module):

    def __init__(self, margin=1.0, imratio=None):
        super(functional_square_loss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.margin = margin
        self.p = imratio
        # https://discuss.pytorch.org/t/valueerror-cant-optimize-a-non-leaf-tensor/21751
        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)  # cuda()
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)  # .cuda()
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(
            self.device)  # .cuda()

    def forward(self, predictions,labels):
        a_coeff, b_coeff, c_coeff, running_loss = 0,0,0,0
        I_pos = torch.where(labels == 1)[0]
        I_neg = torch.where(labels == -1)[0]
        for i in I_pos:
            z_coeff = self.margin - predictions[i]
            a_coeff += 1
            b_coeff += (2)*z_coeff
            c_coeff += z_coeff**2
        for j in I_neg:
            predicted_value = predictions[j]
            running_loss += a_coeff*(predicted_value**2) + b_coeff*predicted_value + c_coeff
        return running_loss