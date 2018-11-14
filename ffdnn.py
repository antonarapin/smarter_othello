import numpy as np

class simple_net:
    
    def __init__(self,
                num_hid,
                num_in,
                num_l,
                num_out,
                lr):
        self.lr = lr
        self.num_hid = num_hid
        self.num_in = num_in
        self.num_l = num_l
        self.num_out = num_out
        # initialize weights as random values in range (-1,1)
        dims = [self.num_in]+[self.num_hid for i in range(self.num_l)]+[self.num_out]
        self.weights = []
        #self.funcs = [(self.sig,self.dsig),(self.relu,self.drelu),(self.sig,self.dsig)]
        self.funcs = [(self.sig,self.dsig)]
        self.funcs += [(self.relu,self.drelu) for x in range(self.num_l-1)]
        self.funcs.append((self.sig,self.dsig))
        for i in range(len(dims)-1):
            self.weights.append(2*np.random.random((dims[i+1],dims[i]))-1)
        
    def sig(self, x):
        """sigmoid"""
        return 1/(1+np.e**(-x))

    def dsig(self, x):
        """sigmoid derivative"""
        return x*(1-x)

    def relu(self, x):
        """linear rectifier"""
        return x*(x>0)

    def drelu(self, x):
        """linear rectifier derivative"""
        return (x>0)*1

    def train(self, x, y):
        """train the network by completing forward and backward paths"""
        inputs = np.array(x, ndmin=2).T
        outputs = [inputs]+self.forward(inputs)
        target = np.array(y, ndmin=2).T
        self.backward(target, outputs)

    def predict(self,x):
        """predict based on trained weights"""
        inputs = np.array(x, ndmin=2).T
        res = self.forward(inputs)
        return res[-1]

    def forward(self, inputs):
        """perform forward pass"""
        outputs = []
        layer_out = inputs
        for l in range(self.num_l+1):
            layer_out = np.dot(self.weights[l],layer_out)
            layer_out = self.funcs[l][0](layer_out)
            outputs.append(layer_out)

        return outputs

    def backward(self, target, outputs):
        """perform backpropagation"""
        layer_err = target - outputs[-1]
        for i in range(len(outputs)-1):
            e_h1_o = layer_err * self.funcs[-(i+1)][1](outputs[-(i+1)])
            self.weights[-(i+1)] += self.lr * np.dot(e_h1_o, outputs[-(i+2)].T)
            layer_err = np.dot(self.weights[-(i+1)].T, layer_err)
