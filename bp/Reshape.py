import numpy as np

class Reshape():

    def __init__(self,ids,shape,ifOutput,preLayer):
        self.name = 'reshape'
        self.ids = ids
        self.toShape = shape
        self.ifOutput = ifOutput
        self.preLayer = preLayer
        self.firstLayer = False
        self.n_nums = shape[0]
        if self.preLayer.name == 'convolution':
            self.n_nums = shape[1]

    def computeForward(self,model):
        if self.preLayer.name == model.layers[self.ids+1].name:
            self.A = self.preLayer.A.reshape(self.toShape)
        else:
            self.A = self.preLayer.A.reshape(self.toShape).T
            # self.n_nums = self.toShape[1]
        # print('Reshape:\n ' + str(self.A))
        # print('-------')

    def computeBackward(self,model):
        self.fromShape = self.preLayer.A.shape
        if self.ifOutput:
            self.dX = 1/model.samples * (self.A - model.y)
        if self.preLayer.name == model.layers[self.ids+1].name:
            self.dX = model.layers[self.ids + 1].dX.reshape(self.fromShape)
        else:
            self.dX = model.layers[self.ids + 1].dX.T.reshape(self.fromShape)

    def update(self,lr):
        pass