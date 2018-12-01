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
        self.l1 = nn.Linear(64,64)
        self.l2 = nn.Linear(64,1)

    def forward(self, x):
        l1o = func.relu(self.l1(x))
        l2o = func.tanh(self.l2(x))

        return l2o 

