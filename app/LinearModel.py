import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
   def __init__(self):
      super().__init__()
      self.fc1 = nn.Linear(x, 1)
   def forward(self, x):
      x = torch.flatten(x,1) #flatten all dimensions execpt batch
      x = self.fc1(x)
      return x