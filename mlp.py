
# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

#Modified by Horace Facey, 2018

import numpy as np

class mlp:
    """ A Multi-Layer Perceptron"""
    
    def __init__(self,numInputs=None,numTargets=None,nhidden=4,beta=1,momentum=0.9,weight1File=None,weight2File=None):
        """ Constructor """
        
        #self.ndata = np.shape(inputs)[0]
        self.nhidden = nhidden

        self.beta = beta
        self.momentum = momentum
        self.outtype = 'linear'
    
        # Initialise network
        if weight1File == None:
            # Set up network size
            if numInputs== None or numTargets==None:
                raise ValueError("If weight files aren't provided, numInputs and numTargets must be given")
            self.nin = numInputs
            self.nout = numTargets
            #initialize weights
            self.weights1 = (np.random.rand(self.nin+1,self.nhidden)-0.5)*2/np.sqrt(self.nin)
            self.weights2 = (np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)
        else:
            self.weights1 = np.loadtxt(weight1File,delimiter=',')
            self.weights2 = np.reshape(np.loadtxt(weight2File,delimiter=','),(-1,1))

        self.bestW1 = None
        self.bestW2 = None
        self.bestErr = float("inf")

    def earlystopping(self,inputs,targets,valid,validtargets,eta,niterations=100):
    
        valid = np.concatenate((valid,-np.ones((np.shape(valid)[0],1))),axis=1)
        
        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000
        
        count = 0
        while (((old_val_error1 - new_val_error) > 0.01) or ((old_val_error2 - old_val_error1)>0.01)):
            self.mlptrain(inputs,targets,eta,niterations)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.mlpfwd(valid)
            new_val_error = 0.5*np.sum((validtargets-validout)**2)
            count+=1
            print("loop:",count,"currError:",new_val_error,"ErrDiff:",old_val_error1 - new_val_error)
                        
        #print("Stopped", new_val_error,old_val_error1, old_val_error2)
        if new_val_error<self.bestErr:
            self.bestErr = new_val_error
            self.bestW1 = np.copy(self.weights1)
            self.bestW2 = np.copy(self.weights2)
        return new_val_error
    	
    def mlptrain(self,inputs,targets,eta,niterations):
        """ Train the thing """    
        # Add the inputs that match the bias node
        self.ndata = np.shape(inputs)[0]
        inputs = np.concatenate((inputs,-np.ones((self.ndata,1))),axis=1)
        change = range(self.ndata)
    
        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))
            
        for n in range(niterations):
    
            self.outputs = self.mlpfwd(inputs)

            error = 0.5*np.sum((self.outputs-targets)**2)
            #if (np.mod(n,100)==0):
                #print("Iteration: ",n, " Error: ",error)    

            # Different types of output neurons
            if self.outtype == 'linear':
            	deltao = (self.outputs-targets)/self.ndata
            elif self.outtype == 'logistic':
            	deltao = self.beta*(self.outputs-targets)*self.outputs*(1.0-self.outputs)
            elif self.outtype == 'softmax':
                deltao = (self.outputs-targets)*(self.outputs*(-self.outputs)+self.outputs)/self.ndata 
            else:
            	print("error")
                
            deltah = self.hidden*self.beta*(1.0-self.hidden)*(np.dot(deltao,np.transpose(self.weights2)))
                      
            updatew1 = eta*(np.dot(np.transpose(inputs),deltah[:,:-1])) + self.momentum*updatew1
            updatew2 = eta*(np.dot(np.transpose(self.hidden),deltao)) + self.momentum*updatew2
            self.weights1 -= updatew1
            self.weights2 -= updatew2
                
            # Randomise order of inputs (not necessary for matrix-based calculation)
            #np.random.shuffle(change)
            #inputs = inputs[change,:]
            #targets = targets[change,:]
            
    def mlpfwd(self,inputs):
        """ Run the network forward """

        self.hidden = np.dot(inputs,self.weights1)
        self.hidden = 1.0/(1.0+np.exp(-self.beta*self.hidden))
        if inputs.ndim==1:
            self.hidden = np.append(self.hidden,-1)
        else:
            self.hidden = np.concatenate((self.hidden,-np.ones((np.shape(inputs)[0],1))),axis=1)

        outputs = np.dot(self.hidden,self.weights2)

        # Different types of output neurons
        if self.outtype == 'linear':
        	return outputs
        elif self.outtype == 'logistic':
            return 1.0/(1.0+np.exp(-self.beta*outputs))
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(outputs),axis=1)*np.ones((1,np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs))/normalisers)
        else:
            print("error")

    def confmat(self,inputs,targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs = self.mlpfwd(inputs)
        nclasses = np.shape(targets)[1]

        if nclasses==1:
            nclasses = 2
            outputs = np.where(outputs>0,1,0)
            targets = np.where(targets>0,1,0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)
        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

        print("Confusion matrix is:")
        print(cm)
        #print("Percentage Correct: ",np.trace(cm)/np.sum(cm)*100)
        return cm,np.trace(cm)/np.sum(cm)*100

    def saveWeights(self,w1file='fstWeights.data',w2file='sndWeights.data'):
        np.savetxt(w1file,self.bestW1,fmt='%.6f',delimiter=',')
        np.savetxt(w2file,self.bestW2,fmt='%.6f',delimiter=',')

    def randomRestart(self,swap=0.1):
        """Randomly reinitializes a fraction of the learned weights,
        attempting to jump from local mimima, and saves the best set"""
        if self.bestErr >= float("inf"):
            return None

        numSwap = int(swap*(self.nin+1))
        if numSwap==0:
            numSwap=1
        swapIdx = []        
        while len(swapIdx)<numSwap:
            idx = np.random.randint(0,self.nin+1) #returns int on interval [low,high)
            if (not idx in swapIdx):
                swapIdx.append(idx)
        for i in swapIdx:
            self.weights1[i] = (np.random.random()-0.5)*2/np.sqrt(self.nin)

        