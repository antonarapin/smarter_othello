import sys,mlp,numpy as np

data = np.loadtxt(sys.argv[1],delimiter=',')
parts = []
partsT = []
n = 5 #number of folds

for i in range(n): #the data is already separated by pos/neg targets, so partitions are even
    part = data[i::n,0:-1] #sections the data into n=5 partitions
    partT = data[i::n,-1:] #with equal number of pos and neg samples

    order = list(range(np.shape(part)[0])) #shuffle the samples in each partition
    np.random.shuffle(order)
    part = part[order,:]
    partT = partT[order,:]

    parts.append(part) #store each fold
    partsT.append(partT)

nin = np.shape(part)[1]
nout = np.shape(partT)[1]
if len(sys.argv)>=4:
    net = mlp.mlp(weight1File=sys.argv[2],weight2File=sys.argv[3])
else:
    net = mlp.mlp(numInputs=nin,numTargets=nout)

for i in range(n):
    train = []
    trainT = []
    for j in range(n):
        if i!=j:
            train.append(parts[j])
            trainT.append(partsT[j])
    train = np.concatenate(tuple(train))
    trainT = np.concatenate(tuple(trainT))

    err = net.earlystopping(train,trainT,parts[i],partsT[i],0.4)
    print("fold:",i,"had error:",err)
    net.confmat(parts[i],partsT[i])
net.saveWeights()
            