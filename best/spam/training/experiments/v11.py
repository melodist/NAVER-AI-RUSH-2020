from spam.spam_classifier.datasets.dataset import Dataset
from spam.spam_classifier.models.AverageModel import AverageModel
from spam.spam_classifier.networks.ResNet50 import frozen_resnet

input_size = (256, 256, 3)
classes = ['normal', 'monotone', 'screenshot', 'unknown']
config = {
    'model': AverageModel,
    'fit_kwargs': {
        'batch_size': 128,
        'epochs_finetune': 50,
        'epochs_full': 10,
        'class_weight': {0: 0.231, 1: 0.247, 2: 0.242, 3: 0.279},
        'debug': False
    },
    'model_kwargs': {
        'network_fn': frozen_resnet,
        'network_kwargs': {
            'input_size': input_size,
            'n_classes': len(classes)
        },
        'dataset_cls': Dataset,
        'dataset_kwargs': {
            'classes': classes,
            'input_size': input_size
        },
    },
}
