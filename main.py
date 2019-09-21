import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
port = 8008

import torch
from main_aid import TBHandler
from mmodel import get_module


if __name__ == "__main__":

    torch.cuda.manual_seed(960301)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    model_name = "DADA"
    tb = TBHandler(model_name)
    param, model = get_module(model_name)

    try:
        model.writer = tb.get_writer()
        tb.star_shell_tb(port)
        model.train_module()
        
    finally:
        tb.kill_shell_tb()
        raise


