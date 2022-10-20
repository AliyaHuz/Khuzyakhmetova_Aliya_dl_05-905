"""Config class"""
from asyncio.windows_events import NULL
import json


class Config:
    """Config class which contains data, train and model hyperparameters"""

    data = NULL
    train = NULL
    model = NULL


    def __init__(self, path):
        self.from_file(path)

    @classmethod
    def from_json(cls, cfg):
        """Creates config from json"""
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.data, params.train, params.model)

    @classmethod
    def from_file(cls, path):
        """Creates config from json file"""
        params = json.load(open(path, 'r'), object_hook=HelperObject)
        cls.data = params.data
        cls.train = params.train
        cls.model = params.model


class HelperObject(object):
    """Helper class to convert json into Python object"""

    def __init__(self, dict_):
        self.__dict__.update(dict_)

    def items(self):
        return self.__dict__.items()