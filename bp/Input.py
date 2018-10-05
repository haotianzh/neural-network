import numpy as np
from bp.Layer import Layer

class Input():

    def __init__(self,x):
        self.name = 'input'
        self.A = x
        self.n_nums = x.shape[0]
        self.firstLayer = True
        return