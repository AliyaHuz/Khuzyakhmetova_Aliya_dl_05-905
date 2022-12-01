from asyncio.windows_events import NULL
from utils.BaseModelConfig import BaseModelConfig

class Data(BaseModelConfig):

    __path = NULL
    __image_size = NULL
    __load_with_info = NULL
    __nrof_classes = NULL
    __shuffle = NULL

    @property
    def path(self):
        return self.__path

    @property
    def image_size(self):
        return self.__image_size

    @property
    def load_with_info(self):
        return self.__load_with_info

    @property
    def nrof_classes(self):
        return self.__nrof_classes

    @property
    def shuffle(self):
        return self.__shuffle

    def _parse_xml_to_datamodel(self,datamodel):
        self.__path = datamodel.find('path').text
        self.__image_size = datamodel.find('image_size').text
        self.__load_with_info = datamodel.find('load_with_info').text
        self.__nrof_classes = datamodel.find('nrof_classes').text
        self.__shuffle = datamodel.find('shuffle').text