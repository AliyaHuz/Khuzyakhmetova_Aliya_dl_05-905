import os
import tensorflow as tf
import numpy as np

from dataloaders.dataloader import DataLoader
from datasets.mnist_dataset import MNISTDataset, DatasetType
from models.base_model import BaseModel
from models.cross_entripy_loss import CrossEntripyLoss
from models.sgd import SGD
from utils.config import Config


class Trainer:
    def __init__(self, cfg):
        self.i = 0
        self.net = BaseModel(cfg['model']['parameters'])
        self.loss = CrossEntripyLoss()
        self.optimizer = SGD(cfg['train']['learning_rate'], self.net, self.loss)
        conf = Config(os.path.abspath("configs/config.json"))
        self.train_ds = MNISTDataset(dataset_type=DatasetType.train, transforms=[], nrof_classes=conf.data.nrof_classes)
        #test_ds = MNISTDataset(dataset_type=DatasetType.train, transforms=[], nrof_classes=conf.data.nrof_classes, ifname='train-images-idx3-ubyte.gz',lfname='train-labels-idx1-ubyte.gz')
        self.train_ds._read_data()
        #test_ds._read_data()
        summary_writer = tf.summary.FileWriter('/tmp/tensorflow_logs/example', graph = tf.get_default_graph())
        self.train_dataloader = DataLoader(self.train_ds, cfg.train.batch_size, None, cfg.train.nrof_epoch)
        #test_dataloader = DataLoader(test_ds, 64, None, 20)

    def fit(self):
        """
        функция обучения нейронной сети, включает в себя цикл по эпохам, в рамках одной эпохи:
        вызов train_epoch;
        вызов evaluate для обучающей и валидационной выборки последовательно
        """
        ...

    def evaluate(self, dataloader,i):
        """
        функция вычисления значения целевой функции и точности
        на всех данных из dataloader, сохранение результатов в tensorboard/mlflow/plotly
        :param dataloader: train_loader/valid_loader/test_loader
        """
        merged_summary_op = tf.summary.merge_all()
        summary_writer.add_summary(merged_summary_op)


    def train_epoch(self):
        """
        функция обучения нейронной сети: для каждого батча из обучающей
        выборки вызывается метод _train_step и логируется значение целевой функции и точности на батче
        """
        self.net.train()
        for batch, labels in self.train_dataloader.batch_generator():
            self._train_step(batch, labels)
        self.net.eval()

    def _train_step(self, batch, labels):
        """
        один шаг обучения нейронной сети, в рамках которого происходит:
        1) вызов forward pass нейронной сети для батча;
        2) вычисление значения целевой функции;
        3) вызов функции minimize для вычисления обратного распространения и обновления весов
        """
        self.i +=1
        logits = self.net(batch)
        batch_loss = self.loss(logits, labels)
        batch_accuracy = np.argmax(logits, 1)
        self.optimizer.minimize()
        tf.summary.scalar('accuracy',batch_accuracy)
        tf.summary.scalar('logits', logits)
        merged_summary_op = tf.summary.merge_all()
        summary_writer.add_summary(merged_summary_op, self.i)
        return batch_loss, batch_accuracy

    def overfit_on_batch(self):
        batch, labels = next(self.train_dataloader.batch_generator())
        for i in range(1000):
            self._train_step(batch, labels)
