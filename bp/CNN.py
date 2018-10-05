from bp.Model import Model
import numpy as np

# 卷积神经网络 架构 还不错。。
x = np.ones([8,2,2,1])
x[(1,3,5,7),:,:,:] = np.zeros([4,2,2,1])
y = np.array([[1, 0, 1, 0, 1, 0, 1, 0]])
model = Model(x, y,inputType='image')
model.addConv(shape=(2,2,1,1),ifOutput=False)
model.addReshape(shape=[8,1])
model.addLayer(10,'sigmoid',False)
model.addLayer(1,'sigmoid',True)
for epoch in range(1000):
    model.train(10)
