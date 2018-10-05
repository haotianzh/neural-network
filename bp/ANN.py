import numpy as np
from bp.Model import Model

# ANN 经典 2 -> 3 -> 6 -> 1
# x = np.linspace(-1,1,20)
x = np.array([[1, 2], [2, 3], [3, 3], [1, 4], [-1, -2], [-1, -1], [-2, -3], [-3, -2]]).T
y = np.array([[0, 0, 0, 0, 1, 1, 1, 1]])
model = Model(x, y,inputType='normal')
model.addLayer(3, 'sigmoid', False)
model.addLayer(6, 'sigmoid', False)
# model.addReshape([6,x.shape[1]])
model.addLayer(1, 'sigmoid', True)
for i in range(1000):
    model.train(10)