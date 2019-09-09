import os
import random
from time import strftime
from subprocess import Popen

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from mmodel import get_module
from mmodel.basic_params import basic_params
from mtrain.watcher import watcher
from torch.utils.tensorboard import SummaryWriter
from mtrain.recorder.info_print import cprint, WARMN, HINTS


TB_CMD = 'tensorboard --logdir "{}" --reload_interval 5 --port 8008'
HINT_FILE_DIR = "Training event file at location: %s"
HINT_TF_ADDR = "Tensorboard running at http:://127.0.0.1:8008 with PID %s"
HINT_RERUN = 'Using "%s" to cheack training process.'

if __name__ == "__main__":

    random.seed(960301)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    model_name = "Center_Loss_Toy"
    param, model = get_module(model_name)

    tb_writer, tb_poc, tb_cmd = None, None, None
    if basic_params.make_record:
        time_stamp = strftime("[%m-%d] %H-%M-%S")
        log_path = os.path.join("records", model_name, time_stamp)
        tb_writer = SummaryWriter(log_dir=log_path, flush_secs=1)
        TB_CMD = TB_CMD.format(log_path)
        tb_poc = Popen(tb_cmd, shell=True)
        cprint(WARMN, HINT_FILE_DIR % log_path)
        cprint(WARMN, HINT_TF_ADDR % tb_pid)

    try:
        model.writer = tb_writer
        model.train_module()
    except:
        raise
    finally:
        model.clean_up()
        if tb_pid is not None:
            tb_poc.kill()
            cprint(HINTS, HINT_RERUN % str(TB_CMD))

