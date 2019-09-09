import itertools

import numpy as np
import torch

from mdata.data_iter import EndlessIter
from mdata.sampler import BalancedSampler
from mdata.dataset.partial import PartialDataset
from mdata.dataset import for_dataset, for_digital_transforms
from mdata.dataset.utils import universal_label_mapping

from torch.utils.data.dataloader import DataLoader

from ..basic_module import TrainableModule
from .toy_params import params

from mground.math_utils import euclidean_dist
from mground.loss_utils import mmd_loss
from torch.nn import functional as F

from functools import partial
import math

from torch.nn.functional import cosine_similarity


class CenterLossToy(TrainableModule):
    def __init__(self):
        self.ce = torch.nn.CrossEntropyLoss()

        super().__init__(params)

    def _prepare_data(self):
        """
            prepare your dataset here
            and return a iterator dic
        """

        trans = for_digital_transforms(is_rgb=False)
        sou_set = for_dataset("MNIST", split="train", transfrom=trans)
        val_set = for_dataset("MNIST", split="test", transfrom=trans)

        _DataLoader = partial(
            DataLoader,
            batch_size=128,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        )
        sou_loader = _DataLoader(sou_set)
        val_loader = _DataLoader(val_set, num_workers=0)

        iters = {
            "train": EndlessIter(sou_loader),
            "valid": EndlessIter(val_loader, max=10)
        }

        return 10, iters

    def _feed_data(self, mode, *args, **kwargs):

        if mode == "train":
            its = self.iters["train"]
            return its.next()
        elif mode == "valid":
            its = self.iters["valid"]
            return its.next()
        raise Exception("feed error!")

    def _regist_networks(self):
        from .networks.lenet import Net
        from .loss import CenterLoss
        return {"N": Net(), "Cet": CenterLoss(num_classes=10, feat_dim=2)}

    def _regist_losses(self):

        optimer = {
            "type": torch.optim.SGD,
            "lr": 0.001,
            "weight_decay": 0.0005,
            "momentum": 0.9,
            "lr_mult": {"Cet": 500},
        }

        self._define_loss(
            "total",
            networks_key=["N", "Cet"],
            optimer=optimer,
        )

        self._define_log(
            "softmax", "cet",
        )
        

    def _train_process(self, datas):

        imgs, labels = datas
        feats, preds = self.N(imgs)
        loss_softmax = self.ce(preds, labels)
        loss_center = self.Cet(feats, labels)
        L = loss_softmax + loss_center

        self._update_losses(
            {
                "total": L,
                "softmax": loss_softmax,
                "cet": loss_center,
            }
        )

    def _eval_process(self, datas):
        imgs, _ = datas
        feats, preds = self.N(imgs)
        props, predcition = torch.max(preds, dim=1)
        return predcition
