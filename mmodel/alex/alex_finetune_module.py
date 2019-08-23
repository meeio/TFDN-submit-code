import itertools

import numpy as np
import torch

from mdata.data_iter import EndlessIter
from mdata.sampler import BalancedSampler
from mdata.dataset.partial import PartialDataset
from mdata.dataset import for_dataset, alex_net_transform
from mdata.dataset.utils import universal_label_mapping

from torch.utils.data.dataloader import DataLoader

from ..basic_module import TrainableModule
from .alex_finetune_params import get_params

from mground.math_utils import euclidean_dist
from mground.loss_utils import mmd_loss
from torch.nn import functional as F

from functools import partial
import math

from torch.nn.functional import cosine_similarity
from torch.nn import CrossEntropyLoss


# get params from defined basic params fucntion
param = get_params()


def dist_based_prediction(pred_pto, center_pto):
    dist = euclidean_dist(pred_pto, center_pto)
    dist = -1 * dist
    pred = torch.nn.functional.log_softmax(dist, dim=1)
    return pred


class AlexFinetune(TrainableModule):
    def __init__(self):


        clses = list(range(31))
        sou_cls = clses[0:31]
        tar_cls = clses[0:31]

        label_mapping = universal_label_mapping(
            sou_cls=sou_cls, tar_cls=tar_cls
        )
        print(label_mapping)

        self.cls_info = {
            "sou_cls": sou_cls,
            "tar_cls": tar_cls,
            "cls_num": len(sou_cls) + 1,
            "mapping": label_mapping,
            "unkown": len(sou_cls),
        }

        self.bce = CrossEntropyLoss()
        self.nll = torch.nn.NLLLoss()

        super().__init__(param)

    def update_cet_ptos(self, ptos):
        if self.cet_ptos is None:
            self.cet_ptos = ptos
        else:
            old_ptos = self.cet_ptos.detach()
            p = cosine_similarity(ptos, old_ptos)
            p = (p ** 2).unsqueeze(1).detach()
            self.cet_ptos = (1 - p) * old_ptos + (p) * ptos
        return self.cet_ptos

    def cls_ptos(self, ptos, targets):
        cls_num = len(self.cls_info["sou_cls"])
        cls_ptos_idx = torch.cat(
            [targets.eq(c).unsqueeze(0) for c in range(cls_num)]
        )
        cls_ptos_cet = torch.cat(
            [
                (ptos[cls_ptos_idx[c]]).mean(dim=0, keepdim=True)
                for c in range(cls_num)
            ]
        )
        return cls_ptos_cet, cls_ptos_idx

    def _prepare_data(self):
        """
            prepare your dataset here
            and return a iterator dic
        """

        t_trans = alex_net_transform(is_train=True)
        sou_set = for_dataset("office31", sub_set="A", transfrom=t_trans)
        tar_set = for_dataset("office31", sub_set="W", transfrom=t_trans)
        v_trans = alex_net_transform(is_train=False)
        val_set = for_dataset("office31", sub_set="W", transfrom=v_trans)

        _ParitalDataset = partial(
            PartialDataset, cls_mapping=self.cls_info["mapping"]
        )
        sou_set = _ParitalDataset(sou_set, self.cls_info["sou_cls"])
        tar_set = _ParitalDataset(tar_set, self.cls_info["tar_cls"])
        val_set = _ParitalDataset(val_set, self.cls_info["tar_cls"])

        _DataLoader = partial(
            DataLoader,
            batch_size=128,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
        )

        sou_loader = _DataLoader(
            sou_set, sampler=BalancedSampler(sou_set.targets)
        )
        tar_loader = _DataLoader(tar_set, shuffle=True)
        val_loader = _DataLoader(val_set, shuffle=True, batch_size=32, drop_last=False)

        iters = {
            "train": {
                "sou_iter": EndlessIter(sou_loader),
                "tar_iter": EndlessIter(tar_loader),
            },
            "valid": {
                "sou_iter": EndlessIter(sou_loader, max=10),
                "val_iter": EndlessIter(val_loader, max=-1),
            },
        }

        return self.cls_info["cls_num"], iters

    def _feed_data(self, mode, *args, **kwargs):

        if mode == "train":
            its = self.iters["train"]
            return its["sou_iter"].next()
        elif mode == "pre_valid":
            its = self.iters["valid"]
            return its["sou_iter"].next()
        elif mode == "valid":
            its = self.iters["valid"]
            imgs_label = its["val_iter"].next()
            if imgs_label is None:
                return imgs_label
            imgs, labels = imgs_label
            return imgs, labels

        raise Exception("feed error!")

    def _regist_networks(self):
        from .networks.alex import AlexFeature, AlexFc
        nets = {"F": AlexFeature(), "C": AlexFc()}
        return nets

    def _regist_losses(self):

        optimer = {
            "type": torch.optim.SGD,
            "lr": 0.001,
            "weight_decay": 0.0005,
            "momentum": 0.9,
            "lr_mult": {"F": 0.1},
        }

        lr_scheduler = {
            "type": torch.optim.lr_scheduler.LambdaLR,
            "lr_lambda": lambda step: (
                (1 + 10 * (step / self.total_steps)) ** (-0.75)
            ),
            "last_epoch": 1,
        }

        self._define_loss(
            "total",
            networks_key=["F", "C"],
            optimer=optimer,
            # decay_op=lr_scheduler,
        )


    def _train_process(self, datas):

        s_imgs, s_labels = datas

        # generate sample prototypes and center prototypes
        labels = self.C(self.F(s_imgs))
        # t_ptos = self.G(self.F(t_imgs))
        L = self.bce(labels, s_labels)

        self._update_losses(
            {
                "total": L,
            }
        )

    def _eval_process(self, datas):
        img, label = datas
        p = self.C(self.F(img))
        props, predcition = torch.max(p, dim=1)
        return predcition

 
