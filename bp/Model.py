import numpy as np
from bp.Layer import Layer
from bp.Input import Input
from bp.Reshape import Reshape
from bp.Convolution import Convolution
import matplotlib.pyplot as plt
class Model:

    def __init__(self,x,y,inputType):
        self.layers = []
        self.ids = 1
        self.x = x
        self.y = y
        # 如果正常输入是 shape[1]  如果是图片input 是shape[0]
        if inputType == 'image':
            self.samples = self.x.shape[0]
        else:
            self.samples = self.x.shape[1]
        self.__addFirstLayer()

    def __addFirstLayer(self):
        input = Input(self.x)
        self.layers.append(input)

    def addReshape(self,shape):
        l = Reshape(ids=self.ids,shape=shape,ifOutput=False,preLayer=self.layers[-1])
        self.ids += 1
        self.layers.append(l)

    def addLayer(self,n_nums,activition,ifOutput):
        l = Layer(ids=self.ids,n_nums=n_nums,activition=activition,input_dim=self.layers[-1].n_nums,if_output=ifOutput,pre_layer=self.layers[-1])
        #l.computeForward()
        self.ids += 1
        self.layers.append(l)

    def addConv(self,shape,ifOutput):
        l = Convolution(ids=self.ids,shape=shape,ifOutput=ifOutput,preLayer=self.layers[-1])
        self.ids += 1
        self.layers.append(l)

    def computeMetrics(self):
        y_hat = np.round(self.layers[-1].A)
        diff = np.abs(y_hat - self.y)
        print(self.layers[-1].A)
        print(1 - (np.sum(diff) / self.samples))

    def train(self,epochs):
        # forward
        for layer in self.layers:
            if layer.firstLayer:
                continue
            layer.computeForward(self)

        # 计算统计量
        self.computeMetrics()

        # backward
        # print(len(self.layers))
        for index in range(len(self.layers)-1,0,-1):
            self.layers[index].computeBackward(self)
            # print('----')
            # print(self.layers[index].dX.shape)

        # update
        for layer in self.layers:
            if layer.firstLayer:
                continue
            layer.update(0.1)

