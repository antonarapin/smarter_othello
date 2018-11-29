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
        self.l2 = nn.Linear(64,64)
        self.l3 = nn.Linear(64,64)
        self.l4 = nn.Linear(64,64)
        self.l5 = nn.Linear(64,64)
        self.l6 = nn.Linear(64,64)
        self.l7 = nn.Linear(64,1)

    def forward(self, x):
        l1o = func.tanh(self.l1(x))
        l2o = func.relu(self.l2(x))
        l3o = func.tanh(self.l3(x))
        l4o = func.relu(self.l4(x))
        l5o = func.tanh(self.l5(x))
        l6o = func.relu(self.l6(x))
        l7o = func.tanh(self.l7(x))

        return l7o 

