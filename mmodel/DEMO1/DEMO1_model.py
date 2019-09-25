from .DEMO1_params import params

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


class DEMO1Model(TrainableModule):
    def __init__(self):
        self.CE = torch.nn.CrossEntropyLoss()
        self.NCE = torch.nn.CrossEntropyLoss(reduction="none")
        self.BCE = torch.nn.BCEWithLogitsLoss()
        self.KL = torch.nn.KLDivLoss()

        shared = [0, 1, 5, 10, 11, 12, 15, 16, 17, 22]
        sou_private = [2, 3, 4, 6, 7, 8, 9, 13, 14, 18]
        tar_private = [19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30]

        self.share = shared
        self.private = sou_private

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
        self.SS = (torch.ones([size, 1], dtype=torch.float) - 0.1).cuda()
        self.TT = (torch.zeros([size, 1], dtype=torch.float) + 0.1).cuda()
        self.threshold = torch.Tensor([0.95]).cuda()
        self.zeros = torch.zeros([36, 512]).cuda()
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
            num_workers=8,
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
            SDomainDis,
            SDisentangler,
            Conver,
        )

        def dy_adv_coeff(iter_num, high=1.0, low=0.0, alpha=10.0):
            iter_num = max(iter_num - 3000, 0)
            return np.float(
                2.0
                * (high - low)
                / (1.0 + np.exp(-alpha * iter_num / self.total_steps))
                - (high - low)
                + low
            )

        return {
            "F": ResnetFeat(),
            "N_d": Disentangler(in_dim=2048, out_dim=1024),
            "N_c": Disentangler(in_dim=2048, out_dim=512),
            "N_dd": SDisentangler(in_dim=1024, out_dim=512),
            "N_dc": SDisentangler(in_dim=1024, out_dim=512),
            "C": ClassPredictor(
                cls_num=self.cls_info["cls_num"],
                adv_coeff_fn=lambda: dy_adv_coeff(self.current_step),
            ),
            "D_dc": SDomainDis(
                in_dim=512,
                adv_coeff_fn=lambda: dy_adv_coeff(self.current_step),
            ),
            "D_dd": SDomainDis(
                in_dim=512,
                adv_coeff_fn=lambda: dy_adv_coeff(self.current_step),
            ),
            "D": DomainDis(
                adv_coeff_fn=lambda: dy_adv_coeff(self.current_step)
            ),
            "M_d": Mine(f=512, s=512),
            "R_d": Reconstructor(),
            "Cr": Conver(
                adv_coeff_fn=lambda: dy_adv_coeff(self.current_step)
            ),
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
            "GlobalDisCls", networks_key=["F", "N_d", "D", "N_c", "C"]
        )

        define_loss(
            "Distangle",
            networks_key=["N_dd", "N_dc", "D_dd", "D_dc", "R_d", "M_d"],
        )

        define_loss("Distangle_cls_dis", networks_key=["C", "N_dc"])

        define_loss("Distangle_cls_adv", networks_key=["N_dd"])

        define_loss("Domain_adv", networks_key=["F", "N_c", "D_dd", "Cr"])

    def reconstruct_loss(self, src, tgt):
        return torch.sum((src - tgt) ** 2) / (src.shape[0] * src.shape[1])

    def get_loss(self, imgs, labels=None, t=True):

        L = dict()

        """ distengled features """
        feats = self.F(imgs)
        feats_d = self.N_d(feats)
        feats_c = self.N_c(feats)
        feats_dd = self.N_dd(feats_d)
        feats_dc = self.N_dc(feats_d)

        """ recon loss """
        rec_d_feats = self.R_d([feats_dd, feats_dc])
        L["rec"] = {"d": self.reconstruct_loss(feats_d, rec_d_feats)}

        """ mutual info loss """
        L["diff"] = {
            "mut": self.M_d.mutual_est(feats_dc, feats_dd),
            "nrom": torch.sum(feats_dd * feats_dc)
            / feats_dd.shape[0],
        }

        """ domain predictions based on different feats """
        # training different domain discriminator
        domain_preds_d = self.D(feats_d)
        domain_preds_dd = self.D_dd(feats_dd)
        domain_preds_dc = self.D_dc(feats_dc)
        # to ensure reconstruct feature is maintain domain information
        domain_preds_d_r = self.D(rec_d_feats)
        # domain adversarial between
        domain_preds_dd_c = self.D_dd(self.Cr(feats_c, adv=True))

        _BCE = partial(self.BCE, target=self.T if t else self.S)
        L["dom"] = {
            "dis_d": _BCE(domain_preds_d),
            "dis_dd": _BCE(domain_preds_dd),
            "dis_dc": _BCE(domain_preds_dc),
            "dis_d_r": _BCE(domain_preds_d_r),
            "adv_dd_c": _BCE(domain_preds_dd_c),
        }

        share_dc_dis = 0
        private_dc_dis = 0

        partial_rec_dc = self.R_d([self.zeros, feats_dc]).detach()
        domain_preds_dc = self.D(partial_rec_dc)
        domain_preds_dc = torch.sigmoid(domain_preds_dc)

        partial_rec_dd = self.R_d([feats_dd, self.zeros]).detach()
        domain_preds_dd = self.D(partial_rec_dd)
        domain_preds_dd = torch.sigmoid(domain_preds_dd)

        """ classify loss """
        if t:
            rdomain_preds_dc = 0.6 - domain_preds_dc
            w = 36 * rdomain_preds_dc / rdomain_preds_dc.sum()
            loss_cls = ent(self.C(feats_c), w)
            loss_cls_dc = None
            loss_cls_dd = ent(self.C(feats_dd))
            loss_cls_adv_dd = -loss_cls_dd
        else:
            w = 36 * domain_preds_dc / domain_preds_dc.sum()
            loss_cls = self.CE(self.C(feats_c), labels)
            loss_cls_dc = torch.mean(self.NCE(self.C(feats_dc), labels)*w)
            loss_cls_dd = None
            loss_cls_adv_dd = None

        L["cls"] = {
            "l": loss_cls,
            "cls_dc": loss_cls_dc,
            "cls_dd": loss_cls_dd,
            "cls_adv_dd": loss_cls_adv_dd,
        }

        if t:
            private_dc_dis = torch.mean(domain_preds_dc[labels == 20])
            share_dc_dis = torch.mean(domain_preds_dc[labels != 20])

            private_dd_dis = torch.mean(domain_preds_dd[labels == 20])
            share_dd_dis = torch.mean(domain_preds_dd[labels != 20])

            private_w = torch.mean(c[labels == 20])
            share_w = torch.mean(c[labels != 20])
        else:
            shared_mask = torch.sum(
                torch.stack([(labels == i) for i in self.share], dim=1),
                dim=1,
            )
            private_mask = 1 - shared_mask

            shared_mask = shared_mask.bool()
            private_mask = private_mask.bool()

            share_dc_dis = torch.mean(domain_preds_dc[shared_mask])
            private_dc_dis = torch.mean(domain_preds_dc[private_mask])

            share_dd_dis = torch.mean(domain_preds_dd[shared_mask])
            private_dd_dis = torch.mean(domain_preds_dd[private_mask])

            share_w = torch.mean(c[shared_mask])
            private_w = torch.mean(c[private_mask])

        L["CET"] = {
            "sh": share_dc_dis,
            "pr": private_dc_dis,
            "sh_w": share_w,
            "pr_w": private_w,
            "sh_dd": share_dd_dis,
            "pr_dd": private_dd_dis,
        }

        return L

    def _train_process(self, datas):

        s_imgs, s_labels, t_imgs, t_labels = datas

        engage_recon = True if self.current_step > 1200 else False
        engage_entropy = True if self.current_step > 1200 else False
        engage_adv = True if self.current_step > 1900 else False

        Ls = self.get_loss(s_imgs, s_labels, t=False)
        Lt = self.get_loss(t_imgs, t_labels, t=True)

        def L(f, s, c=1, d=None):
            if d is None:
                return c * (Ls[f][s] + Lt[f][s]) / 2
            elif d == "S":
                return c * (Ls[f][s])
            elif d == "T":
                return c * (Lt[f][s])

        # disentangle losses
        L_diff_mut = L("diff", "mut", 0.01)
        L_diff_norm = L("diff", "norm", 0.0001)
        # recon losses
        L_rec_d = L("rec", "d")
        L_dis_d_r = L("dom", "dis_d_r") if engage_recon else 0
        # discriminator losses
        L_dis_d = L("dom", "dis_d")
        L_dis_dd = L("dom", "dis_dd")
        L_dis_dc = L("dom", "dis_dc")
        L_adv_d_c = L("dom", "adv_d_c") if engage_adv else 0
        # classifier losss
        L_cls = L("cls", "l") if engage_entropy else L("cls", "l", "S")
        L_cls_dc = L("cls", "cls_dc", "S")
        L_cls_dd = L("cls", "cls_dd", "T")
        L_cls_adv_dd = L("cls", "cls_adv_dd", "T")

        self._update_losses(
            {
                "GlobalDisCls": L_dis_d + L_cls,
                "Distangle": L_dis_dd
                + L_dis_dc
                + L_diff_mut
                + L_diff_norm
                + L_dis_d_r,
                "Distangle_cls_dis": L_cls_dc + L_cls_dd,
                "Distangle_cls_adv": L_cls_adv_dd,
                "Domain_adv": L_adv_d_c,
            }
        )

        self._update_logs(
            {
                "DomDis/d": L_dis_d,
                "DomDis/dd": L_dis_dd,
                "DomDis/dc": L_dis_dc,
                "DomDis/d_r": L_dis_d_r,
                "DomDis/d_adv_c": L_adv_d_c,
                "Cls/c": L_cls,
                "Cls/dc": L_cls_dc,
                "Cls/dd": L_cls_dd,
                "diff_mut": L_diff_mut,
                "diff_norm": L_diff_norm,
            }
        )

        sh = Lt["CET"]["sh"]
        shd = Lt["CET"]["sh_dd"]
        sh_w = Lt["CET"]["sh_w"]
        self._update_logs({"CET_DC/t_share": sh})
        self._update_logs({"CET_DD/t_share": shd})
        self._update_logs({"CET_W/share_w": sh_w})
        pr = Lt["CET"]["pr"]
        prd = Lt["CET"]["pr_dd"]
        pr_w = Lt["CET"]["pr_w"]
        if pr == pr:
            self._update_logs({"CET_DC/t_private": pr})
            self._update_logs({"CET_DD/t_private": prd})
            self._update_logs({"CET_W/private_w": pr_w})

        sh = Ls["CET"]["sh"]
        shd = Ls["CET"]["sh_dd"]
        # sh_w = Ls["CET"]["sh_w"]
        self._update_logs({"CET_DC/s_share": sh})
        self._update_logs({"CET_DD/s_share": shd})
        # self._update_logs({"CET_S/share_w": sh_w})
        pr = Ls["CET"]["pr"]
        prd = Ls["CET"]["pr_dd"]
        # sh_w = Ls["CET"]["pr_dd"]
        # pr_w = Ls["CET"]["pr_w"]
        if pr == pr:
            self._update_logs({"CET_DC/s_private": pr})
            self._update_logs({"CET_DD/s_private": prd})
            # self._update_logs({"CET_S/private_w": pr_w})

    def _eval_process(self, datas):
        imgs, l = datas
        feats = self.F(imgs)

        feats_dc = self.N_dc(self.N_d(feats))
        partial_rec_feats = self.R_d([self.zeros, feats_dc])
        domain_preds_d = self.D(partial_rec_feats)
        domain_preds_d = torch.sigmoid(domain_preds_d).squeeze()

        t = torch.max(domain_preds_d[l != 20], dim=-1)[0]

        feats_di = self.N_c(feats)
        preds = F.softmax(self.C(feats_di), dim=-1)
        props, predcition = torch.max(preds, dim=1)
        predcition[props < 0.9] = self.cls_info["cls_num"]
        return predcition
