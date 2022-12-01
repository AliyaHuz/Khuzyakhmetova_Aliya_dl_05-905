from asyncio.windows_events import NULL
from utils.BaseModelConfig import BaseModelConfig

class Model(BaseModelConfig):
    __input = []
    __up_stack = {}
    __activation_function = NULL
    __output = NULL

    @property
    def input(self):
        return self.__input

    @property
    def up_stack(self):
        return self.__up_stack

    @property
    def activation_function(self):
        return self.__activation_function

    @property
    def output(self):
        return self.__output

    def __init__(self,dataModel):
        super().__init__(dataModel)

    def _parse_xml_to_datamodel(self,datamodel):
        self.__activation_function = datamodel.find('activation_function').text
        self.__output = datamodel.find('output').text

        for up_stacks in datamodel.find('up_stack'):
            self.__up_stack.update({up_stacks.tag : up_stacks.text})
        for inputs in datamodel.findall('input'):
            self.__input.append(inputs.text)