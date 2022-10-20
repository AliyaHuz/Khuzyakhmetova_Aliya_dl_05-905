import gzip
import os
from ast import List
from typing import List
import json
from datasets.dataset import BaseDataSet
from enum import Enum
from typing import Callable
import numpy as np
import idx2numpy

"""здесь пишем класс по считыванию данных"""

class DatasetType(Enum):
    train = 'train'
    valid = 'valid'
    test = 'test'

class MNISTDataset(BaseDataSet):
    def __init__(self, dataset_type: DatasetType, transforms: List[Callable], nrof_classes: int,
                 data_path = os.path.abspath("data/KMNIST/row"), ifname='train-images-idx3-ubyte.gz', lfname='train-labels-idx1-ubyte.gz'):
        self.__dataset_path = data_path
        self.__images = []
        self.__labels = []
        self.__image_file_name = ifname
        self.__label_file_name = lfname
        self.__dataset_type = dataset_type
        self.__transforms = transforms
        self.__nrof_classes = nrof_classes

    def _read_labels(self):
        with gzip.open(os.path.join(self.__dataset_path,self.__label_file_name), 'r') as f:
            _ = int.from_bytes(f.read(4), 'big')  # magic number
            _ = int.from_bytes(f.read(4), 'big')  # number of labels
            # rest is the label data, each label is stored as unsigned byte
            # label values are 0 to 9
            label_data = f.read()
            labels = np.frombuffer(label_data, dtype=np.uint8)
            return labels

    def _read_images(self):
        with gzip.open(os.path.join(self.__dataset_path,self.__image_file_name), 'r') as f:
            _ = int.from_bytes(f.read(4), 'big')  # magic number
            image_count = int.from_bytes(f.read(4), 'big')  # number of images
            row_count = int.from_bytes(f.read(4), 'big')  # row count
            column_count = int.from_bytes(f.read(4), 'big')  # column count
            # rest is the image pixel data, each pixel is stored as an unsigned byte
            # pixel values are 0 to 255
            image_data = f.read()
            images = np.frombuffer(image_data, dtype=np.uint8) \
                .reshape((image_count, row_count, column_count))

            return images

    def _read_data(self):
        self.__labels = self._read_labels()
        self.__images = self._read_images()

        self.show_statistics()

    def __len__(self):
        """
        :return: размер выборки
        """
        return len(self.__images)

    def one_hot_labels(self, label):
        """
        для 10 классов метка 5-> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        :param label: метка класса
        :return: one-hot encoding вектор
        """
        if abs(label) > self.__nrof_classes:
            raise ValueError(f'Метки {label} не существует!')
        return [1 if el == label else 0 for el in range(self.__nrof_classes)]

    def __getitem__(self, idx):
        """
        :param idx: индекс элемента в выборке
        :return: preprocessed image and label
        """
        images = self.__images[idx]
        labels = self.__labels[idx]
        for transform in self.__transforms:
            images = transform(images)
        return images, labels

    def show_statistics(self):
        """
        Необходимо вывести количество элементов в датасете, количество классов и количество элементов в каждом классе
        """
        unique, counts = np.unique(self.__labels, return_counts=True)
        print(f'Dataset - {self.__dataset_type.name} \nNumber of elements in dataset is {self.__len__()} \n Number of classes is {self.__nrof_classes} \n{dict(zip(unique, counts))}\n')
