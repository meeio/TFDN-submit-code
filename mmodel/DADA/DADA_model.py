from .DADA_params import params

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

from mground.loss_utils import mmd_loss
from torch.nn import functional as F

from functools import partial
import math

from mdata.dataset.utils import universal_label_mapping
from mdata.dataset.partial import PartialDataset

import torch.nn.functional as F
from ..utils.math.entropy import ent


class DADAModule(TrainableModule):
    def __init__(self):
        self.CE = torch.nn.CrossEntropyLoss()
        self.BCE = torch.nn.BCEWithLogitsLoss()

        shared = [0, 1, 5, 10, 11, 12, 15, 16, 17, 22]
        sou_private = [2, 3, 4, 6, 7, 8, 9, 13, 14, 18]
        tar_private = [19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30]

        clses = list(range(31))
        sou_cls = clses
        tar_cls = clses
        self.cls_info = {
            "sou_cls": sou_cls,
            "tar_cls": tar_cls,
            "cls_num": len(sou_cls),
            "mapping": universal_label_mapping(sou_cls, tar_cls),
        }

        size = params.batch_size
        self.S = torch.ones([size, 1], dtype=torch.float).cuda()
        self.T = torch.zeros([size, 1], dtype=torch.float).cuda()
        self.ST = self.T + 0.5 
        self.threshold = torch.Tensor([0.5]).cuda()
        super().__init__(params)

    def _prepare_data(self):

        sou_set = for_dataset(
            "OFFICE31", split="A", transfrom=resnet_transform(is_train=True)
        )
        tar_set = for_dataset(
            "OFFICE31", split="W", transfrom=resnet_transform(is_train=True)
        )
        val_set = for_dataset(
            "OFFICE31", split="W", transfrom=resnet_transform(is_train=False)
        )

        # _ParitalDataset = partial(
        #     PartialDataset, cls_mapping=self.cls_info["mapping"]
        # )

        # val_set = _ParitalDataset(val_set, self.cls_info["tar_cls"])
        # sou_set = _ParitalDataset(sou_set, self.cls_info["sou_cls"])
        # tar_set = _ParitalDataset(tar_set, self.cls_info["tar_cls"])

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

        return self.cls_info["cls_num"], data_feeding_fn

    def _regist_networks(self):
        from .networks.nets import (
            ResnetFeat,
            Disentangler,
            SDisentangler,
            DomainDis,
            SDomainDis,
            ClassPredictor,
            Reconstructor,
            Mine,
            Conver,
        )

        def dy_adv_coeff(iter_num, high=1.0, low=0.0, alpha=10.0):
            return np.float(
                2.0
                * (high - low)
                / (1.0 + np.exp(-alpha * iter_num / self.total_steps))
                - (high - low)
                + low
            )
        
        self.dy_adv = dy_adv_coeff

        return {
            "F": ResnetFeat(),
            "C": ClassPredictor(cls_num=self.cls_info["cls_num"]),
            "N_dr": Disentangler(in_dim=2048, out_dim=2048),
            "N_drr": SDisentangler(in_dim=2048, out_dim=1024),
            "N_di": Disentangler(in_dim=2048, out_dim=2048),
            "D": DomainDis(in_dim=2048),
            "D_rr": SDomainDis(
                in_dim=1024,
                # adv_coeff_fn=lambda: dy_adv_coeff(self.current_step)
            ),
            "M": Mine(),
            "R": Reconstructor(),
            "Cr": Conver(
                adv_coeff_fn=lambda: dy_adv_coeff(self.current_step)
            )
        }

    def _regist_losses(self):
        def dy_lr_coeff(iter_num, alpha=10, power=0.75):
            return np.float(
                (1 + alpha * (iter_num / self.total_steps)) ** (-power)
            )

        optimer = {
            "type": torch.optim.SGD,
            "lr": 0.01,
            "weight_decay": 0.0005,
            "momentum": 0.9,
            "lr_mult": {"F": 0.1},
        }

        decay_op = {
            "type": torch.optim.lr_scheduler.LambdaLR,
            "last_epoch": 0,
            "lr_lambda": lambda step: dy_lr_coeff(step),
        }

        define_loss = partial(
            self._define_loss, optimer=optimer, decay_op=decay_op
        )

        define_loss(
            "Total_loss", networks_key=["F", "N_di", "N_dr", "D", "M", "R", "C"]
        )

        define_loss(
            "Dis", networks_key=["N_drr", "D_rr"]
        )

        define_loss(
            "Adv", networks_key=["F", "N_di", "Cr", "D_rr"]
        )
        


    def mutual_info_estimate(self, x, y, y_):
        joint, marginal = self.M(x, y), self.M(x, y_)
        return torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)))

    def reconstruct_loss(self, src, tgt):
        return torch.sum((src - tgt) ** 2) / (src.shape[0] * src.shape[1])

    def get_loss(self, imgs, labels=None):
        feats = self.F(imgs)
        feats_di = self.N_di(feats)
        feats_dr = self.N_dr(feats)
        feats_drr = self.N_drr(feats_dr)

        L = dict()

        # recon loss
        rec_feats = self.R(feats_di, feats_dr)
        L['rec'] = {'f': self.reconstruct_loss(feats, rec_feats)}

        # mutual info loss
        shuffile_idx = torch.randperm(feats_di.shape[0])
        shuffled_feats_di = feats_di[shuffile_idx]
        loss_mutual = self.mutual_info_estimate(
            feats_dr, feats_di, shuffled_feats_di
        )
        L['diff'] = {'mut': loss_mutual}

        # adv training
        domain_preds_dr = self.D(feats_dr)
        domain_preds_drr = self.D_rr(feats_drr)
        domain_preds_di = self.D_rr(self.Cr(feats_di, adv=True))

        if labels is None:
            domain_target = self.T
            loss_cls = ent(self.C(feats_di))
            loss_dr_cls = ent(self.C(feats_dr))
            loss_dr_adv = - loss_dr_cls
        else:
            domain_target = self.S
            loss_cls = self.CE(self.C(feats_di, adv=False), labels)
            preds_dr_cls = self.C(feats_dr, adv=False)
            loss_dr_cls = self.CE(preds_dr_cls, labels)
            loss_dr_adv = - ent(preds_dr_cls)

        L['cls'] = {'cls': loss_cls}

        loss_adv = self.BCE(domain_preds_di, domain_target)
        loss_domain = self.BCE(domain_preds_dr, domain_target)
        loss_disen = self.BCE(domain_preds_drr, domain_target)

        L['dom'] = {
            "dis": loss_domain,
            "adv": loss_adv,
            "den": loss_disen
        }

        return L

    def _train_process(self, datas):

        s_imgs, s_labels, t_imgs, _ = datas

        Ls = self.get_loss(s_imgs, s_labels)
        Lt = self.get_loss(t_imgs)

        def L(f, s, c=[1, 1]):
            c = c if isinstance(c, list) else [c] * 2
            return (c[0] * Ls[f][s] + c[1] * Lt[f][s]) / 2

        L_diff_mut = L("diff", "mut", 0.0001)
        L_rec = L("rec", "f", 0.01)
        L_cls = L("cls", "cls", c=[1,0])
        L_dis = L("dom", "dis")
        L_dsen = L("dom", "den")
        L_dis_adv = L("dom", "adv", c=1 if self.current_step>1500 else 0)

        self._update_losses(
            {
                "Total_loss": L_rec + L_diff_mut + L_dis + L_cls,
                "Dis": L_dsen,
                "Adv": L_dis_adv,
            }
        )

        self._update_logs(
            {
                "Domain/dis": L_dis,
                "Domain/adv": L_dis_adv,
                "Domain/sen": L_dsen,
            }
        )

    def _eval_process(self, datas):
        imgs, _ = datas
        feats = self.F(imgs)
        feats_di = self.N_di(feats)
        preds = self.C(feats_di)
        props, predcition = torch.max(preds, dim=1)
        return predcition
