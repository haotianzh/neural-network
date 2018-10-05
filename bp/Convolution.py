import numpy as np

class Convolution():


    def initalizeParams(self):
        self.W = np.random.randn(self.shape[0],self.shape[1],self.shape[2],self.shape[3])
        self.b = np.zeros([1,self.ksize])
        # 初始化一个 w shape的矩阵，在convAdd中使用
        # self.wConvAdd = np.zeros(self.windowWidth,self.windowHeight,self.ksize)
        # for i in range(self.windowWidth):
        #     for j in range(self.windowHeight):
        #         self.wConvAdd[i,j,:] = 1


    def __init__(self,ids,shape,ifOutput,preLayer):
        self.name = 'convolution'
        self.firstLayer = False
        self.ids = ids
        self.shape = shape
        # self.ksize = ksize
        self.samples = preLayer.A.shape[0]
        self.ifOutput = ifOutput
        self.preLayer = preLayer
        self.inputWidth = self.preLayer.A.shape[1]
        self.inputHeight = self.preLayer.A.shape[2]
        self.windowWidth = self.shape[0]
        self.windowHeight = self.shape[1]
        self.outputWidth = self.inputWidth - self.windowWidth + 1
        self.outputHeight = self.inputHeight - self.windowHeight + 1
        self.ksize = self.shape[3]
        # print ("input dx,dy:(%d,%d),output dx,dy:(%d,%d),kenerl size:%d"%(self.inputWidth,
        #                                                                   self.inputHeight,
        #                                                                   self.outputWidth,
        #                                                                   self.outputHeight,
        #                                                                   self.ksize))
        self.initalizeParams()

    def convAdd(self,sameMatrix):
        # 对于同维度小矩阵和权值矩阵进行向量叠加
        result = 0.0
        for i in range(self.windowWidth):
            for j in range(self.windowHeight):
                result += sameMatrix[:,i,j,i,j,:]
                # print("result" + str(result.shape))
        return result

    def computeForward(self,model):
        # print("begin")
        temp_x = np.dot(self.preLayer.A,self.W) + self.b
        # print('temo_x:' + str(temp_x.shape))
        self.A = np.zeros([self.samples,self.outputWidth, self.outputHeight, self.ksize])
        for i in range(self.inputWidth - self.windowWidth + 1):
            for j in range(self.inputHeight-self.windowHeight + 1):
                sameMatrix = temp_x[:,i:i + self.windowWidth,j:j + self.windowHeight,:,:,:]
                self.A[:,i,j] = self.convAdd(sameMatrix=sameMatrix)
            # print(self.A)
        # print('forward done!')
        del(temp_x)
    def computeBackward(self,model):

        def computeDWAndDXAndDb():
            dW = np.zeros_like(self.W)
            dX = np.zeros_like(self.preLayer.A)
            db = np.zeros_like(self.b)
            # 遍历整个dZ 依次累加 dW dX
            for i in range(dZ.shape[1]):
                for j in range(dZ.shape[2]):
                    dz = dZ[:,i,j,:] #  8 x 10
                    for m in range(self.windowWidth):
                        for n in range(self.windowHeight):
                            dW[m,n,:,:] += np.dot(self.preLayer.A[:,i+m,j+n,:].T,dz) # 100 x10 = 100 x 8 x 8 x 10
                            dX[:,i+m,j+n,:] += np.dot(dz,self.W[m,n,:,:].T) # 8 x 100 = 8 x 10 x 10 x 100
                            db += np.dot(np.ones([1,self.samples]),dz)
            return dW,dX,db
        dZ = model.layers[self.ids+1].dX
        self.dW,self.dX,self.db = computeDWAndDXAndDb()

    def update(self,lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db







