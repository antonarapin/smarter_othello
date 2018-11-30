import sys,mlp,numpy as np

n = 5 #number of folds
data = np.loadtxt("endgameResults.data",delimiter=',')
#for s in range(10000,100001,10000):
for s in range(5,6):
    #print("____________________________________")
    print("now learning with",s*0.1,"random restart")
    #data = allData[:s,:]
    #data  = np.loadtxt("end"+str(s)+".data",delimiter=',')
    parts = []
    partsT = []

    for i in range(n):
        part = data[i::n,0:-1] #sections the data into n=5 partitions
        partT = data[i::n,-1:] 

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
        cm,acc = net.confmat(parts[i],partsT[i])
        print("fold:",i,"had error:",err,"and acc:",acc)
        net.randomRestart(0.1*s)
    f1 = "fWt0."+str(s)+"RR.data"
    f2 = "sWt0."+str(s)+"RR.data"
    net.saveWeights(w1file=f1,w2file=f2)
            