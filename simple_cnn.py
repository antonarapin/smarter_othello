import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as func
import torch.optim as optim
import random
import time

class Othello_CNN(nn.Module):
    
    def __init__(self):
        super(Othello_CNN, self).__init__()
        self.c1 = nn.Conv2d(1,1,kernel_size=3)
        self.l1 = nn.Linear(36,1)

    def forward(self, x):
        o1 = self.c1(x)
        fo1 = func.tanh(o1)
        o = fo1.view(1,36)
        output = func.tanh(self.l1(o))
        return output

