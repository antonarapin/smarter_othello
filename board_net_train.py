import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as func
import torch.optim as optim
import random
import time
from board_weights_net import Othello_CNN

def cnn_train(lr):
    net = Othello_CNN()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)
    fname = "bnet_model"+str(lr)+".pt"
    lossfile = open("lossfile_"+fname,'w+')

    # read data
    data = np.loadtxt("endgameResults.data",delimiter=",") 
    # just go through all data unfiltered
    ep = 0
    start = time.time()
    for t in range(10):
        np.random.shuffle(data)
        for data_pt in data:
            ep+=1
            x = data_pt[:-1]
            y = data_pt[-1]
            #print(y)
            if y<0:
                y=-1
            elif y>0:
                y=1
            target = torch.FloatTensor(np.array([y]))
            inputs = torch.FloatTensor(x)
            optimizer.zero_grad()
            output = net(inputs)
            #print(target,output)
            loss = criterion(output, target)
            lossfile.write(str(float(loss.data))+"\n")
            if ep%1000==0:
                print(ep,target.data[0],output.data[0],loss.data[0])
            #time.sleep(1)
            loss.backward()
            optimizer.step()
    end = time.time()
    print("TIME ELAPSED:",end-start)
    print("savig to " + fname + "...")
    torch.save(net,fname)
    print("saved")
cnn_train(0.05)
# cnn_train(0.5)
# cnn_train(0.6)
# cnn_train(0.7)
