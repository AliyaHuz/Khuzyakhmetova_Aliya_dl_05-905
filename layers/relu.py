from layers.base_layer_class import BaseLayerClass
from numpy import *
from layers.registry_utils import REGISTRY_TYPE

@REGISTRY_TYPE.register_module
class ReLU(BaseLayerClass):
    def __call__(self, x):
        self.x = x
        return np.max(0, x)

    def get_grad(self):
        self.grads = np.zeros_like(self.x)
        self.grads[self.x>0] = 1
        return self.grads

    def backward(self, dy):
        return dy*self.grads
