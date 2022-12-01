import json
from pathlib import Path

CFG = {
    "data": {
        "path": '../data/KMNIST/row',
        "image_file_name": "train-images-idx3-ubyte",
        "label_file_name": "train-labels-idx1-ubyte",
        "dataset_type": "train",
        "transforms": ['Normalize', 'View'],
        "nrof_classes": 10,
        "shuffle": True
    },
    "train": {
        "batch_size": 64,
        "nrof_epoch": 20,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy", "balanced_accuracy"],
        "learning_rate": 0.01
    },
    "model": {
        "parameters":[('FullyConnected', {'input_size': 784, 'output_size': 128}), ('ReLU', {}),
              ('FullyConnected', {'input_size': 128, 'output_size': 10})]
    }
}
with open('config.json', 'w') as write_file:
    json.dump(CFG,write_file)

