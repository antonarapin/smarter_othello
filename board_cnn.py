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
        self.c2 = nn.Conv2d(1,1,kernel_size=2)
        self.mp = nn.MaxPool2d(2,stride=2)
        self.l = nn.Linear(4,1)

    def forward(self, x):
        c1out = self.c1(x)
        sigout = func.tanh(c1out)
        mpout = self.mp(sigout)
        c2out = self.c2(mpout)
        o = c2out.view(1,4)
        linout = self.l(o)
        output = func.tanh(linout)
        return output

