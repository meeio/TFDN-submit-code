import itertools

import numpy as np
import torch

from mdata.data_iter import inf_iter
from mdata.sampler import BalancedSampler
from mdata.dataset.partial import PartialDataset
from mdata.dataset import for_dataset, resnet_transform
from mdata.dataset.utils import universal_label_mapping

from torch.utils.data.dataloader import DataLoader

from ..basic_module import TrainableModule
from .demo_params import params

from mground.math_utils import euclidean_dist
from mground.loss_utils import mmd_loss
from torch.nn import functional as F

from functools import partial
import math

from torch.nn.functional import cosine_similarity
from mground.math_utils import euclidean_dist

from mdata.dataset.utils import universal_label_mapping
from mdata.dataset.partial import PartialDataset

import torch.nn.functional as F


class DemoModule(TrainableModule):
    def __init__(self):
        self.CE = torch.nn.CrossEntropyLoss()
        self.BCE = torch.nn.BCELoss()
        
        clses = list(range(65))
        sou_cls = clses[0:10] + clses[10:15]
        tar_cls = clses[0:10] + clses[15:65]
        self.cls_info = {
            "sou_cls": sou_cls,
            "tar_cls": tar_cls,
            "cls_num": len(sou_cls),
            "mapping": universal_label_mapping(sou_cls, tar_cls),
        }

        size = params.batch_size
        self.S = torch.ones([size, 1], dtype=torch.float).cuda()
        self.T = torch.zeros([size, 1], dtype=torch.float).cuda()
        self.threshold = torch.Tensor([0.5]).cuda()
        super().__init__(params)

    def _prepare_data(self):

        sou_set = for_dataset("OFFICEHOME", split="Ar",
                              transfrom=resnet_transform(is_train=True))
        tar_set = for_dataset("OFFICEHOME", split="Pr",
                              transfrom=resnet_transform(is_train=True))
        val_set = for_dataset("OFFICEHOME", split="Pr",
                              transfrom=resnet_transform(is_train=False))


        _ParitalDataset = partial(
            PartialDataset, cls_mapping=self.cls_info["mapping"]
        )

        val_set = _ParitalDataset(val_set, self.cls_info["tar_cls"])
        sou_set = _ParitalDataset(sou_set, self.cls_info["sou_cls"])
        tar_set = _ParitalDataset(tar_set, self.cls_info["tar_cls"])

        _DataLoader = partial(
            DataLoader,
            batch_size=params.batch_size,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
        )

        sou_iter = inf_iter(_DataLoader(sou_set))
        tar_iter = inf_iter(_DataLoader(tar_set))
        val_iter = inf_iter(_DataLoader(val_set), with_end=True)

        def data_feeding_fn(mode):
            if mode == "train":
                s_img, s_label = next(sou_iter)
                t_img, t_label = next(tar_iter)
                return s_img, s_label, t_img, t_label
            elif mode == "valid":
                return next(val_iter)

        return self.cls_info["cls_num"] + 1, data_feeding_fn

    def _regist_networks(self):
        from .networks.resnet import ResnetFeat, ResnetCls, DisNet
        from .center_loss import CenterLoss

        def dy_adv_coeff(iter_num, high=1.0, low=0.0, alpha=10.0):
            return np.float(
                2.0 * (high - low) /
                (1.0 + np.exp(-alpha * iter_num / self.total_steps))
                - (high - low)
                + low
            )


        return {
            "F": ResnetFeat(),
            "C": ResnetCls(out_dim=self.cls_info['cls_num']),
            "D": DisNet(
                in_feat_dim=2048,
                adv_coeff_fn=lambda: dy_adv_coeff(self.current_step)),
            # "Cet": CenterLoss(num_classes=31, feat_dim=256),
        }

    def _regist_losses(self):

        def dy_lr_coeff(
            iter_num, alpha=10, power=0.75
        ):
            return np.float((1 + alpha * (iter_num / self.total_steps)) ** (-power))

        optimer = {
            "type": torch.optim.SGD,
            "lr": 0.01,
            "weight_decay": 0.0005,
            "momentum": 0.9,
            "lr_mult": {"F": 0.1, "C": 1, "D":1},
        }

        decay_op = {
            "type": torch.optim.lr_scheduler.LambdaLR,
            "last_epoch": 0,
            "lr_lambda": lambda step: dy_lr_coeff(step),
        }

        self._define_loss(
            "total",
            networks_key=["F", "C", "D"],
            optimer=optimer,
            decay_op=decay_op,
        )

        self._define_log("softmax", "adv")

    def _train_process(self, datas):

        s_imgs, s_labels, t_imgs, labels = datas


        s_feats = self.F(s_imgs)
        _, s_preds = self.C(s_feats)
        s_domain = self.D(s_feats)

        t_feats = self.F(t_imgs)
        _, t_preds = self.C(t_feats)
        t_domain = self.D(t_feats)

        L_softmax = self.CE(s_preds, s_labels)
        L_adv = self.BCE(s_domain, self.S) + self.BCE(t_domain, self.T)
        L_adv = L_adv/2
        L = L_softmax + L_adv

        self._update_losses(
            {
                "total": L,
                "softmax": L_softmax,
                "adv": L_adv,
            }
        )

    def _eval_process(self, datas):
        imgs, _ = datas

        feats = self.F(imgs)
        _, preds = self.C(feats)
        preds = F.softmax(preds, dim=-1)

        props, predcition = torch.max(preds, dim=1)
        predcition[props<self.threshold] = self.cls_info['cls_num']
        return predcition
