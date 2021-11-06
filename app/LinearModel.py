import torch
import torch.nn as nn

class LinearModel(nn.Module):
   def __init__(self):
      super(LinearModel, self).__init__()
      self.fc1 = nn.Linear(3072, 1)

   def forward(self, x):
      x = torch.flatten(x, 1) # flatten all dimensions execpt batch
      x = self.fc1(x)
      return x