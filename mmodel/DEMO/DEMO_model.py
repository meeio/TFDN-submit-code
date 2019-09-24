from .DEMO_params import params

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


class DEMOModel(TrainableModule):
    def __init__(self):
        self.CE = torch.nn.CrossEntropyLoss()
        self.BCE = torch.nn.BCEWithLogitsLoss()

        shared = [0, 1, 5, 10, 11, 12, 15, 16, 17, 22]
        sou_private = [2, 3, 4, 6, 7, 8, 9, 13, 14, 18]
        tar_private = [19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30]

        self.cls_info = {
            "sou_cls": shared + sou_private,
            "tar_cls": shared + tar_private,
            "cls_num": len(shared) + len(sou_private),
            "mapping": universal_label_mapping(
                shared + sou_private, shared + tar_private
            ),
        }

        size = params.batch_size
        self.S = torch.ones([size, 1], dtype=torch.float).cuda()
        self.T = torch.zeros([size, 1], dtype=torch.float).cuda()
        self.threshold = torch.Tensor([0.7]).cuda()
        self.zeros = torch.zeros([36,2048]).cuda()
        super().__init__(params)

    def _prepare_data(self):

        sou_set = for_dataset(
            "OFFICE31",
            split="A",
            transfrom=resnet_transform(is_train=True),
        )
        tar_set = for_dataset(
            "OFFICE31",
            split="W",
            transfrom=resnet_transform(is_train=True),
        )
        val_set = for_dataset(
            "OFFICE31",
            split="W",
            transfrom=resnet_transform(is_train=False),
        )

        _ParitalDataset = partial(
            PartialDataset, ncls_mapping=self.cls_info["mapping"]
        )

        sou_set = _ParitalDataset(sou_set, self.cls_info["sou_cls"])
        tar_set = _ParitalDataset(tar_set, self.cls_info["tar_cls"])
        val_set = _ParitalDataset(val_set, self.cls_info["tar_cls"])

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
        from .networks.nets import (
            ResnetFeat,
            Disentangler,
            DomainDis,
            ClassPredictor,
            Reconstructor,
            Mine,
        )

        def dy_adv_coeff(iter_num, high=1.0, low=0.0, alpha=10.0):
            return np.float(
                2.0
                * (high - low)
                / (1.0 + np.exp(-alpha * iter_num / self.total_steps))
                - (high - low)
                + low
            )

        return {
            "F": ResnetFeat(),
            "N_d": Disentangler(),
            "N_dd": Disentangler(),
            "N_dc": Disentangler(),
            "N_di": Disentangler(),
            "C": ClassPredictor(
                cls_num=self.cls_info["cls_num"],
                adv_coeff_fn=lambda: dy_adv_coeff(self.current_step),
            ),
            "D_dc": DomainDis(
                adv_coeff_fn=lambda: dy_adv_coeff(self.current_step)
            ),
            "D_dd": DomainDis(
                adv_coeff_fn=lambda: dy_adv_coeff(self.current_step)
            ),
            "D": DomainDis(
                adv_coeff_fn=lambda: dy_adv_coeff(self.current_step)
            ),
            "M": Mine(),
            "R_f": Reconstructor(indim=3),
            "R_d": Reconstructor(),
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
            "Total_loss",
            networks_key=[
                "F",
                "N_d",
                "N_dd",
                "N_dc",
                "N_di",
                "D_dd",
                "D_dc",
                "D",
                "M",
                "R_d",
                "R_f",
                "C",
            ],
        )

        define_loss("DomainRelevent/Classify", networks_key=["C"])
        define_loss("DomainRelevent/Adv", networks_key=["F", "N_dd"])

        define_loss("DomainDistangle/Dis", networks_key=["D_dd"])
        define_loss("DomainDistangle/Adv", networks_key=["F", "N_dc"])

    def mutual_info_estimate(self, x, y, y_):
        joint, marginal = self.M(x, y), self.M(x, y_)
        return torch.mean(joint) - torch.log(
            torch.mean(torch.exp(marginal))
        )

    def reconstruct_loss(self, src, tgt):
        return torch.sum((src - tgt) ** 2) / (src.shape[0] * src.shape[1])

    def get_loss(self, imgs, labels=None, t=True):

        feats = self.F(imgs)

        """ distengled features """
        feats_d = self.N_d(feats)
        feats_dd = self.N_dd(feats)
        feats_dc = self.N_dc(feats)
        feats_di = self.N_di(feats)

        """ recon loss """
        # recon backbone features
        rec_feats = self.R_f([feats_di, feats_dd, feats_dc])
        loss_rec_f = self.reconstruct_loss(feats, rec_feats)
        # recon domain relevent features
        rec_d_feats = self.R_d([feats_dd, feats_dc])
        loss_rec_f_d = self.reconstruct_loss(feats_d, rec_d_feats)
        loss_rec = loss_rec_f + loss_rec_f_d

        """ mutual info loss """
        shuffile_idx = torch.randperm(feats_di.shape[0])
        # mutual info between cls_feature and domain_relevent_features
        shuffle_f_di = feats_di[shuffile_idx]
        loss_mutual_di = self.mutual_info_estimate(
            feats_dd, feats_di, shuffle_f_di
        )
        # mutual info between domain_relevent and domain_cls features
        shuffle_f_dc = feats_dc[shuffile_idx]
        loss_mutual_dc = self.mutual_info_estimate(
            feats_dd, feats_dc, shuffle_f_dc
        )
        loss_mutual = loss_mutual_dc + loss_mutual_di

        """ domain predictions based on different feats """
        domain_preds_o = self.D(feats_d, adv=False)
        domain_preds_r = self.D(rec_d_feats, adv=False)
        domain_preds_dd = self.D_dd(feats_dd, adv=False)
        domain_preds_dc = self.D_dc(feats_dc, adv=False)
        domain_preds_dd_di = self.D_dd(feats_di, adv=True)
        domain_preds_dd_dc = self.D_dd(feats_dc)

        BCE = partial(
            self.BCE, target=self.T if t else self.S
        )

        domain_dis_d_loss_o = BCE(domain_preds_o)
        domain_dis_d_loss_r = BCE(domain_preds_r)
        domain_dis_d_loss = domain_dis_d_loss_o+0.3*domain_dis_d_loss_r
        domain_dis_dr_loss = BCE(domain_preds_dd)
        domain_dis_dc_loss = BCE(domain_preds_dc)

        domain_adv_dd_di_loss = BCE(domain_preds_dd_di)
        domain_dis_dd_dc_loss = BCE(domain_preds_dd_dc)
        domain_adv_dd_dc_loss = - ent(domain_preds_dd_dc)

        """ classify loss """
        unknown_pro = 0
        known_pro = 0
        if t:
            loss_cls = ent(self.C(feats_di))
            loss_dr_cls = ent(self.C(feats_dd))
            loss_dr_adv = -loss_dr_cls

            partial_rec_feats = self.R_d([self.zeros, feats_dc])
            domain_preds_d = self.D(partial_rec_feats, adv=False)
            unknown_pro = torch.mean(domain_preds_d[labels==20])
            known_pro = torch.mean(domain_preds_d[labels!=20])
        else:
            loss_cls = self.CE(self.C(feats_di), labels)
            dd_preds_cls = self.C(feats_dd)
            loss_dr_cls = self.CE(dd_preds_cls, labels)
            loss_dr_adv = -ent(dd_preds_cls)

        return {
            "Rec": {
                "T": loss_rec_f + loss_rec_f_d,
                "rec_f": loss_rec_f,
                "rec_d": loss_rec_f_d,
            },
            "Mut": loss_mutual,
            "D_dis": {
                "d": domain_dis_d_loss,
                "dd": domain_dis_dr_loss,
                "dc": domain_dis_dc_loss,
                "T": domain_dis_d_loss
                + domain_dis_dr_loss
                + domain_dis_dc_loss,
            },
            "D_adv": {
                "dd_di_cls_adv": domain_adv_dd_di_loss,
                "dd_dc_cls": domain_dis_dd_dc_loss,
                "dd_dc_adv": domain_adv_dd_dc_loss,
                # "T": domain_adv_dd_di_loss,
            },
            "C_cls": {
                "di_cls": loss_cls,
                "dd_cls": loss_dr_cls,
                "dd_adv": loss_dr_adv,
            },
            "add": {
                "unkown": unknown_pro,
                "known": known_pro,
            }
        }

    def _train_process(self, datas):

        s_imgs, s_labels, t_imgs, t_labels = datas

        Ls = self.get_loss(s_imgs, s_labels, t=False)
        Lt = self.get_loss(t_imgs, t_labels, t=True)

        L_rec = (Ls["Rec"]["T"] + Lt["Rec"]["T"]) / 2
        L_mutual = (Ls["Mut"] + Lt["Mut"]) / 2
        L_d_dis = (Ls["D_dis"]["T"] + Lt["D_dis"]["T"]) / 2
        L_d_adv = (Ls["D_adv"]["dd_di_cls_adv"] + Lt["D_adv"]["dd_di_cls_adv"]) / 2
        L_cls = Ls["C_cls"]["di_cls"] + Lt["C_cls"]["di_cls"]

        L_dd_cls = (Ls["C_cls"]["dd_cls"] + Ls["C_cls"]["dd_cls"]) / 2
        L_dd_adv = (Ls["C_cls"]["dd_adv"] + Ls["C_cls"]["dd_adv"]) / 2

        L_d_dd_dc_dis = (Ls["D_adv"]["dd_dc_cls"] + Ls["D_adv"]["dd_dc_cls"]) / 2
        L_d_dd_dc_adv = (Ls["D_adv"]["dd_dc_adv"] + Ls["D_adv"]["dd_dc_adv"]) / 2

        L_total = (
            L_rec + 0.0005 * L_mutual + L_d_dis + L_d_adv + L_cls
        )

        self._update_losses(
            {
                "Total_loss": L_total,
                "DomainRelevent/Classify": L_dd_cls,
                "DomainRelevent/Adv": L_dd_adv,
                "DomainDistangle/Dis": L_d_dd_dc_dis,
                "DomainDistangle/Adv": L_d_dd_dc_adv,
            }
        )

        self._update_logs(
            {
                "SourceClassify/di": Ls["C_cls"]["di_cls"],
                "SourceClassify/dd": Ls["C_cls"]["dd_cls"],
                "TargetClassify/di_entropy": Lt["C_cls"]["di_cls"],
                "TargetClassify/dr_entropy": Lt["C_cls"]["dd_cls"],
                "Discriminator/d_dis": Ls["D_dis"]["d"] + Lt["D_dis"]["d"],
                "Discriminator/dd": Ls["D_dis"]["dd"] + Lt["D_dis"]["dd"],
                "Discriminator/dc": Ls["D_dis"]["dc"] + Lt["D_dis"]["dc"],
                "DDAdv/non-adv": Ls["D_dis"]["dd"] + Lt["D_dis"]["dd"],
                "DDAdv/di-adv": Ls["D_adv"]["dd_di_cls_adv"]
                + Lt["D_adv"]["dd_di_cls_adv"],
                "DDAdv/dc-adv": Ls["D_adv"]["dd_dc_cls"]
                + Lt["D_adv"]["dd_dc_cls"],
                "DD_CAdv/source_entropy": -Ls["C_cls"]["dd_adv"],
                "DD_CAdv/target_entropy": -Lt["C_cls"]["dd_adv"],
                "Reconstruct/rec_f": Ls["Rec"]["rec_f"] + Lt["Rec"]["rec_f"],
                "Reconstruct/rec_d": Ls["Rec"]["rec_d"] + Lt["Rec"]["rec_d"],
                "MutualInfomation": L_mutual,
            }
        )

        unkp = Lt["add"]["unkown"]
        kp = Lt["add"]["known"]
        self._update_logs(
            {
                "ADD/known": torch.sigmoid(kp)
            }
        )
        if unkp == unkp:
            self._update_logs(
                {
                    "ADD/un_known": torch.sigmoid(unkp)
                }
            )

    def _eval_process(self, datas):
        imgs, _ = datas
        feats = self.F(imgs)
        feats_di = self.N_di(feats)
        preds = F.softmax(self.C(feats_di), dim=-1)
        props, predcition = torch.max(preds, dim=1)
        predcition[props < self.threshold] = self.cls_info["cls_num"]
        return predcition
