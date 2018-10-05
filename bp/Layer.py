import numpy as np


class Layer():

    def __init__(self,ids,n_nums,input_dim,pre_layer,if_output,activition=''):
        self.name = 'dense'
        self.ids = ids
        self.n_nums = n_nums
        self.pre_layer = pre_layer
        self.firstLayer = False
        if activition == None or activition == '':
            self.activition = activition
        self.input_dim = input_dim
        self.if_output = if_output
        self.initalizeParams()


    def initalizeParams(self):
        self.W = np.random.randn(self.n_nums,self.input_dim)
        self.b = np.zeros([self.n_nums,1])
        # self.dZ = None
        # self.dW = None
        # self.dX = None
        # self.db = None
        # self.Z = None
        # self.A = None
        return

    def sigmoid(self,x):
        y = 1 / (1 + np.exp(-x))
        return y

    def dervSigmoid(self,x):
        return np.multiply(self.sigmoid(x),1-self.sigmoid(x))

    def computeBackward(self,model):
        if self.if_output:
            self.dZ = 1/model.samples * (self.A - model.y)
        else:
            self.dZ = np.multiply(model.layers[self.ids+1].dX, self.dervSigmoid(self.Z))
        self.dW = np.dot(self.dZ,self.pre_layer.A.T)
        self.db = np.dot(self.dZ,np.ones([model.samples,1]))
        self.dX = np.dot(self.dW.T,self.dZ)


    def computeForward(self,model):
        # print('compute')
        self.Z = np.dot(self.W,self.pre_layer.A) + self.b
        self.A = self.sigmoid(self.Z)

    def update(self,lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db
