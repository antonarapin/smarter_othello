import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as func
import torch.optim as optim
import random

class Othello_CNN(nn.Module):
    
    def __init__(self):
        super(Othello_CNN, self).__init__()
        self.c1 = nn.Conv2d(1,1,kernel_size=3)
        self.c2 = nn.Conv2d(1,1,kernel_size=2)
        self.mp = nn.MaxPool2d(2,stride=2)
        self.l = nn.Linear(4,1)

    def forward(self, x):
        c1out = self.c1(x)
        sigout = func.sigmoid(c1out)
        mpout = self.mp(sigout)
        c2out = self.c2(mpout)
        o = c2out.view(1,4)
        linout = self.l(o)
        output = func.sigmoid(linout)
        return output


# initialize the network, create the loss criterion and weights update optimizer object
net = Othello_CNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.05)

# read data
data = np.loadtxt("results.data",delimiter=",") 
# just go through all data unfiltered
for data_pt in data:
    x = data[0][:-1].reshape((1,1,8,8))
    y = data[0][-1]
    target = torch.FloatTensor(np.array([[y]]))
    inputs = torch.FloatTensor(x)
    optimizer.zero_grad()
    output = net(inputs)
    loss = criterion(output, target)
    print(loss)
    loss.backward()
    optimizer.step()


