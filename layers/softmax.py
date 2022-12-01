import numpy as np
from layers.registry_utils import REGISTRY_TYPE
from layers.base_layer_class import BaseLayerClass


@REGISTRY_TYPE.register_module
class Softmax(BaseLayerClass):
    def __init__(self):
        pass

    def __call__(self, x, phase):
        self.x = x
        out = np.exp(x)
        return out/np.sum(out)

    def get_grad(self):
        self.grads = np.zeros_like(self.x)
        self.grads[self.x>0] = 1

    def backward(self, dy):
        return dy*self.grads