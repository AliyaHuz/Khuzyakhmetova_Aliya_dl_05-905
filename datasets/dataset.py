from abc import ABC, abstractmethod

class BaseDataSet(ABC):
    @abstractmethod
    def _read_data(self):
        pass
    @abstractmethod
    def __len__(self):
        pass
    @abstractmethod
    def __getitem__(self):
        pass
    @abstractmethod
    def show_statistics(self):
        pass