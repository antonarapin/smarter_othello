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
        self.c1 = nn.Conv2d(1,4,kernel_size=3,padding=1)
        self.c2 = nn.Conv2d(4,4,kernel_size=3,padding=1)
        self.c3 = nn.Conv2d(4,4,kernel_size=2,stride=2)
        self.c4 = nn.Conv2d(4,1,kernel_size=3)
        self.l1 = nn.Linear(4,1)

    def forward(self, x):
        o1 = self.c1(x)
        fo1 = func.tanh(o1)
        o2 = self.c2(fo1)
        fo2 = func.tanh(o2)
        o3 = self.c3(fo2)
        fo3 = func.tanh(o3)
        o4 = self.c4(fo3)
        fo4 = func.tanh(o4)
        o = fo4.view(1,4)
        output = func.tanh(self.l1(o))
        return output

