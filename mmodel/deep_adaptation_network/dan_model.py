import itertools

import numpy as np
import torch

from mdata.data_iter import EndlessIter
from mdata.sampler.balanced_sampler import BalancedSampler
from mdata import for_dataset, for_digital_transforms

from torch.utils.data.dataloader import DataLoader

from ..basic_module import TrainableModule
from .params import get_params

from .networks.networks import LeNetEncoder

from mground.math_utils import euclidean_dist
from torch.nn import functional as F
from mmodel.utils.MMD import MMD

# get params from defined basic params fucntion
param = get_params()


class DeepAdaptationNetworks(TrainableModule):
    def __init__(self):
        super(DeepAdaptationNetworks, self).__init__(param)
        self.CE = torch.nn.CrossEntropyLoss()
        self._all_ready()
    

    def _prepare_data(self):
        """
            prepare your dataset here
            and return a iterator dic
        """

        trans = for_digital_transforms(is_rgb=False)
        sou_set, s_info = for_dataset("mnist", split="train", transfrom=trans)
        tar_set, _ = for_dataset("svhn", split="train", transfrom=trans)
        val_set, _ = for_dataset("svhn", split="test", transfrom=trans)
        data_info = {"cls_num": len(torch.unique(s_info['labels']))}

        sou_loader = DataLoader(sou_set, batch_size=128,shuffle=True, drop_last=True)
        tar_loader = DataLoader(tar_set, batch_size=128,shuffle=True,drop_last=True)
        val_loader = DataLoader(val_set, batch_size=128,drop_last=True)

        iters = {
            "train": {
                "sou_iter": EndlessIter(sou_loader),
                "tar_iter": EndlessIter(tar_loader),
            },
            "valid": EndlessIter(val_loader),
        }

        return data_info, iters

    def _feed_data(self, mode, *args, **kwargs):

        assert mode in ["train", "valid"]
        its = self.iters[mode]
        if mode == "train":
            return its['sou_iter'].next() + its['tar_iter'].next()
        else:
            return its.next(need_end=True)
        

    def _regist_networks(self):
        net = LeNetEncoder()
        return {"N": net}

    def _regist_losses(self):

        optimer = {
            "type": torch.optim.SGD,
            "lr": 0.01,
            "weight_decay": 0.001,
            "momentum": 0.9,
            # "nesterov": True,
        }

        lr_scheduler = {
            "type": torch.optim.lr_scheduler.StepLR,
            "step_size": self.total_steps / 3,
        }

        self._define_loss(
            "total",
            networks_key=["N"],
            optimer=optimer,
            decay_op=lr_scheduler,
        )

        self._define_log('cls_loss','mmd_loss', group='train')
        

    def _train_process(self, datas):
        s_imgs, s_labels, t_imgs, _ = datas

        s_feats, s_preds = self.N(s_imgs)
        t_feats, t_preds = self.N(t_imgs)

        cls_loss = self.CE(s_preds, s_labels)
        mmd_loss = MMD(s_feats, t_feats)

        self._update_losses({
            "total": cls_loss + mmd_loss,
            "cls_loss": cls_loss,
            "mmd_loss": mmd_loss,
        })
    

    def _eval_process(self, datas):
        imgs, labels = datas
        _, preds = self.N(imgs)
        preds = torch.max(preds, dim=1)[1]
        return preds

    