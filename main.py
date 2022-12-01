import os
from asyncio.windows_events import NULL

import numpy as np
from dataloaders.dataloader import DataLoader
from datasets.mnist_dataset import MNISTDataset, DatasetType
from utils.Train import Trainer
from utils.config import Config

def run():
    conf = Config(os.path.abspath("configs/config.json"))
    trainer = Trainer(conf)
    trainer.overfit_on_batch()
    #tensorboard --logdir='Log_Dir'

if __name__ == '__main__':
    run()
    """
    train_ds = MNISTDataset(dataset_type= DatasetType.train, transforms=[], nrof_classes=conf.data.nrof_classes)
    test_ds = MNISTDataset(dataset_type= DatasetType.train, transforms=[], nrof_classes=conf.data.nrof_classes,
                           ifname='train-images-idx3-ubyte.gz',
                           lfname='train-labels-idx1-ubyte.gz')
    train_ds._read_data()
    test_ds._read_data()

    train_dataloader = DataLoader(train_ds, 64, None, 20)
    test_dataloader = DataLoader(test_ds, 64, None, 20)
    next(train_dataloader.batch_generator())
    train_dataloader.show_batch()
    """