import numpy as np

from layers.base_layer_class import BaseLayerClass
import numpy as np
import math
from models.registry import Registry


REGISTRY_TYPE = Registry('Layers')

@REGISTRY_TYPE.register_module
class FullyConnectedLayer(BaseLayerClass):
    def __init__(self, input_size, output_size):
        self.bias = np.zeros(output_size)
        self.weight = np.random.normal(0, math.sqrt(2/output_size), size = (input_size, output_size))

    def __call__(self, x, phase):
        self.x = x
        return np.dot(self.x, self.weight) + self.bias


    @property
    def trainable(self):
        return True

    def get_grad(self):
        df_dx = self.weight
        df_dw = self.x
        df_db = 1
        self.grads = [df_dx, df_dw, df_db]
        return self.grads

    def backward(self, dy):  #добавить усренение
        self.get_grad()
        self.x_grad = np.dot(self.grads[0],dy)
        self.weight_grad = np.dot(self.grads[1],dy)
        self.bias_grad = self.grads[2]*dy

    def update_weights(self, update_func):
        self.weight = self.weight - update_func(self.weight_grad)