from abc import ABC, abstractmethod

from layers.registry_utils import REGISTRY_TYPE
from utils.config import Config


class BaseModel(object):
    def __init__(self, cfg):
        self.params = []
        for pars in cfg:
            self.params.append(REGISTRY_TYPE.get(*cfg))

    def __call__(self, input):
        """
        цикл по всем слоям нейронной сети и вызов метода forward pass (__call__)
        :param input: батч значений предыдущего слоя
        :return: значения текущего слоя
        """
        for layer in self.params:
            input = layer(input, self.phase)
        return input

    def get_parameters(self):
        """
        получение dict, где в качестве key — название слоя, в качестве value — значения обучаемых параметров слоя
        :return:
        """

    def dump_model(self, path):
        """
        функция сохранения параметров нейронной сети в pickle-файл
        :param path: путь, куда сохранять модель
        """

    def load_weights(self, path):
        """
        функция считывания параметров нейронной сети из pickle-файла
        :param path: путь, откуда считывать модель
        """


    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def build(self):
        pass

    def train(self):
        self.phase = 'train'

    def eval(self):
        self.phase = 'eval'

#class BaseModel(ABC):  def __init__(self, cfg): self.config = Config.from_json(cfg)

