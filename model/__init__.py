from .alexnet import *
from .vgg import *
from .nin import *
from .resnet import *
from .inception_v1 import *
from .inception_v2 import *


def get_model(config):
    return globals()[config.architecture](config.num_classes).to(config.device)
