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


# initialize the network, create the loss criterion and weights update optimizer object
"""net = Othello_CNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.09)

# read data
data = np.loadtxt("endgameResults.data",delimiter=",") 
# just go through all data unfiltered
ep = 0
dpts = {}
start = time.time()
for t in range(3):
    np.random.shuffle(data)
    for data_pt in data:
        ep+=1
        x = data_pt[:-1].reshape((1,1,8,8))
        y = data_pt[-1]
        #print(y)
        if y not in dpts.keys():
            dpts[y] = 1
        else:
            dpts[y] = dpts[y]+1
        target = torch.FloatTensor(np.array([[y]]))
        inputs = torch.FloatTensor(x)
        optimizer.zero_grad()
        output = net(inputs)
        #print(target,output)
        loss = criterion(output, target)
        if ep%1000==0:
            print(ep,target.data[0],output.data[0],loss.data[0])
        #time.sleep(1)
        loss.backward()
        optimizer.step()
end = time.time()
print("TIME ELAPSED:",end-start)
print("savig to \'cnn_model1.pt\'...")
torch.save(net,'cnn_model1.pt')
print("saved")"""
