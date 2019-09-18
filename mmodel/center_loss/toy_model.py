import itertools

import numpy as np
import torch

from mdata.data_iter import EndlessIter
from mdata.sampler import BalancedSampler
from mdata.dataset.partial import PartialDataset
from mdata.dataset import for_dataset, resnet_transform
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
from mground.math_utils import euclidean_dist

def dist_based_prediction(pred_pto, center_pto):
    dist = euclidean_dist(pred_pto, center_pto, x_wise=True)
    dist = -1 * dist
    pred = torch.nn.functional.log_softmax(dist, dim=1)
    return pred

class CenterLossToy(TrainableModule):
    def __init__(self):

        self.ce = torch.nn.CrossEntropyLoss()
        super().__init__(params)

    def _prepare_data(self):
        """
                prepare your dataset here
                and return a iterator dic
        """

        sou_set = for_dataset("Office31", split="A",
                              transfrom=resnet_transform(is_train=True))
        val_set = for_dataset("Office31", split="A",
                              transfrom=resnet_transform(is_train=True))

        _DataLoader = partial(
            DataLoader,
            batch_size=70,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
        )
        sou_loader = _DataLoader(sou_set)
        val_loader = _DataLoader(val_set, num_workers=0)

        iters = {
            "train": EndlessIter(sou_loader),
            "valid": EndlessIter(val_loader, max=10)
        }

        def data_feeding_fn(mode):
            if mode == "train":
                its = iters["train"]
                return its.next()
            elif mode == "valid":
                its = iters["valid"]
                return its.next()
            raise Exception("feed error!")

        return 31, data_feeding_fn


    def _regist_networks(self):
        from .networks.resnet import ResnetFeat, ResnetCls
        from .loss import CenterLoss
        return {
            "F": ResnetFeat(),
            "C": ResnetCls(),
            "Cet": CenterLoss(num_classes=31, feat_dim=256),
        }

    def _regist_losses(self):

        optimer = {
            "type": torch.optim.SGD,
            "lr": 0.001,
            "weight_decay": 0.0005,
            "momentum": 0.9,
            "lr_mult": {"F": 0.1, "C": 1, "Cet": 100},
        }

        self._define_loss(
            "total",
            networks_key=["F", "C", "Cet"],
            optimer=optimer,
        )

        self._define_log("softmax", "cet")

    def _train_process(self, datas):

        imgs, labels = datas
        feats = self.F(imgs)
        ptos, preds = self.C(feats)
        loss_softmax = self.ce(preds, labels)
        loss_center = self.Cet(ptos, labels)
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
        feats = self.F(imgs)
        ptos, _ = self.C(feats)
        preds = dist_based_prediction(ptos, self.Cet.centers)
        props, predcition = torch.max(preds, dim=1)
        return predcition
