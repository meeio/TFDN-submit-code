import torch

from mmodel import get_module
from mtrain.watcher import watcher
import os

import random

if __name__ == "__main__":

    random.seed(960301)

    if True:
        torch.backends.cudnn.benchmark = True
        name = "TPN"

        param, A = get_module(name)
        A.train_module()
    else:
        log_to_tbfile = os.getcwd() + "/tb"
        os.system("tensorboard --logdir={}".format(log_to_tbfile))

 