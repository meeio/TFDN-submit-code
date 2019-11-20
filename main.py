import os

confs = [["0", 8006], ["1", 8007], ["2", 8008], ["3", 8009]]
conf = confs[1]

os.environ["CUDA_VISIBLE_DEVICES"] = conf[0]
port = conf[1]

import torch
from main_aid import TBHandler
from mmodel import get_module


if __name__ == "__main__":

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(960301)
    torch.cuda.manual_seed_all(960301)

    model_name = "DEMO1"
    tb = TBHandler(model_name)
    param, model = get_module(model_name)

    try:
        model.writer = tb.get_writer()
        tb.star_shell_tb(port)
        model.train_module()

    finally:
        tb.kill_shell_tb()
        raise

