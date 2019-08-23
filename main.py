import torch

from mmodel import get_module
from mtrain.watcher import watcher
import os

# import random

# if __name__ == "__main__":

#     # random.seed(960301)
#     torch.cuda.empty_cache()

#     if True:
#         torch.backends.cudnn.benchmark = True
#         name = "TPN"

#         param, A = get_module(name)
#         A.train_module()
#     else:
#         log_to_tbfile = os.getcwd() + "/tb"
#         os.system("tensorboard --logdir={}".format(log_to_tbfile))


# t = torch.tensor([[1, 2], [3, 4]])
# a = torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]]))

x = torch.rand(5, 2)
print(x)
a = torch.zeros(3, 2).scatter_add_(
    1, torch.tensor([0,1,2]), x
)
print(a)
