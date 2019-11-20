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
        self.NCE = torch.nn.CrossEntropyLoss(reduction='none')
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
        )

        def dy_adv_coeff(iter_num, high=1.0, low=0.0, alpha=10.0):
            iter_num = max(iter_num - 1900, 0)
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
            "GlobalDisCls",
            networks_key=["F", "N_d", "D", "N_c", "C"],
        )

        define_loss(
            "Distangle",
            networks_key=["N_dd", "N_dc", "D_dd", "D_dc", "R_d", "M_d"],
        )

        define_loss("Distangle_cls_dis", networks_key=["C", "N_dc"])

        define_loss("Distangle_cls_adv", networks_key=["N_dd"])

        define_loss("Domain_adv", networks_key=["F", "N_c", "D_dd"])

    def reconstruct_loss(self, src, tgt):
        return torch.sum((src - tgt) ** 2) / (src.shape[0] * src.shape[1])

    def get_loss(self, imgs, labels=None, t=True):

        feats = self.F(imgs)

        L = dict()

        """ distengled features """
        feats_d = self.N_d(feats)
        feats_c = self.N_c(feats)
        feats_dd = self.N_dd(feats_d)
        feats_dc = self.N_dc(feats_d)

        """ recon loss """
        # recon backbone features
        # rec_feats = self.R_f([feats_d, feats_c])
        # loss_rec_f = self.reconstruct_loss(feats, rec_feats)
        # recon domain relevent features
        rec_d_feats = self.R_d([feats_dd, feats_dc])
        loss_rec_f_d = self.reconstruct_loss(feats_d, rec_d_feats)
        # save loss
        # L["rec"] = {"f": loss_rec_f, "d": loss_rec_f_d}
        L["rec"] = {"d": loss_rec_f_d}

        """ mutual info loss """
        # loss_mut_f = self.M_f.mutual_est(feats_d, feats_c)
        loss_mut_d = self.M_d.mutual_est(feats_dc, feats_dd)

        loss_diff = torch.sum(feats_dd*feats_dc) / 36

        # save loss
        L["mut"] = {
            # "d": loss_mut_f,
            "d": loss_mut_d,
            "dd_dc_diff": loss_diff,
        }

        """ domain predictions based on different feats """
        domain_preds_d = self.D(feats_d, adv=False)
        domain_preds_dc = self.D_dc(feats_dc, adv=False)
        domain_preds_dd = self.D_dd(feats_dd, adv=False)
        domain_preds_d_r = self.D(rec_d_feats, adv=False)
        domain_preds_d_c = self.D_dd(feats_c, adv=True)
        domain_preds_dc_c = self.D_dc(feats_c, adv=False)

        _BCE = partial(self.BCE, target=self.T if t else self.S)
        loss_domain_dis_dc = _BCE(domain_preds_dc)
        loss_domain_dis_dd = _BCE(domain_preds_dd)
        loss_domain_adv_d_c = _BCE(domain_preds_d_c)
        loss_domain_dis_d = _BCE(domain_preds_d)
        loss_domain_dis_d_r = _BCE(domain_preds_d_r)
        loss_domain_dis_dc_c = _BCE(domain_preds_dc_c)
        # save loss
        L["dom"] = {
            "dis_d": loss_domain_dis_d,
            "dis_dd": loss_domain_dis_dd,
            "dis_dc": loss_domain_dis_dc,
            "dis_d_r": loss_domain_dis_d_r,
            "adv_d_c": loss_domain_adv_d_c,
            "dis_dc_c": loss_domain_dis_dc_c,
        }

        share_dc_dis = 0
        private_dc_dis = 0

        partial_rec_dc = self.R_d([self.zeros, feats_dc]).detach()
        domain_preds_dc = self.D(partial_rec_dc)
        domain_preds_dc = torch.sigmoid(domain_preds_dc)

        partial_rec_dd = self.R_d([feats_dd, self.zeros]).detach()
        domain_preds_dd = self.D(partial_rec_dd)
        domain_preds_dd= torch.sigmoid(domain_preds_dd)

        """ classify loss """
        if t:
            rdomain_preds_dc = 1- domain_preds_dc
            c =  rdomain_preds_dc / rdomain_preds_dc.sum()
            c = c * 36
            loss_cls = ent(self.C(feats_c), c)
            loss_cls_dc = ent(self.C(feats_dc))
            loss_cls_dd = ent(self.C(feats_dd))
            loss_cls_adv_dd = -loss_cls_dd

            if self.current_step < 1800:
                loss_cls = 0
            loss_cls_dc = 0

        else:
            loss_cls = self.CE(self.C(feats_c), labels)
            c = domain_preds_dc / domain_preds_dc.sum()
            c = c * 36
            loss_cls_dc = torch.mean(self.CE(self.C(feats_dc), labels)) 
            loss_cls_dd = torch.mean(self.CE(self.C(feats_dd), labels)) 
            loss_cls_adv_dd = -ent(self.C(feats_dd))
            loss_cls_dd = 0
            loss_cls_adv_dd = 0

        L["cls"] = {
            "cls": loss_cls,
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

        L["CET"] = {"sh": share_dc_dis, "pr": private_dc_dis, "sh_w":share_w, "pr_w": private_w, "sh_dd": share_dd_dis, "pr_dd": private_dd_dis}

        return L

    def _train_process(self, datas):

        s_imgs, s_labels, t_imgs, t_labels = datas

        Ls = self.get_loss(s_imgs, s_labels, t=False)
        Lt = self.get_loss(t_imgs, t_labels, t=True)

        def L(f, s, C=1):
            return C * (Ls[f][s] + Lt[f][s]) / 2

        # recs
        # L_rec_f = L("rec", "f", 0.01)
        L_rec_d = L("rec", "d", 0.01)
        # muts
        # L_mut_f = L("mut", "f", 0.0001)
        L_mut_d = L("mut", "d", 0.0001)
        L_dd_dc_diff = L("mut", "dd_dc_diff", 0.001)
        # doms
        L_dis_d = L("dom", "dis_d")
        L_dis_d_r = L("dom", "dis_d_r")
        L_dis_dd = L("dom", "dis_dd")
        L_dis_dc = L("dom", "dis_dc")
        L_adv_d_c = L("dom", "adv_d_c")
        # clses
        L_cls = L("cls", "cls")
        L_cls_dc = L("cls", "cls_dc")
        L_cls_dd = L("cls", "cls_dd")
        L_cls_adv_dd = L("cls", "cls_adv_dd")

        r = 1 if self.current_step > 1500 else 0
        a = 1 if self.current_step > 3000 else 0

        self._update_losses(
            {
                "GlobalDisCls": L_dis_d + L_cls,
                "Distangle": L_dis_dd
                + L_dis_dc
                + L_mut_d
                + L_dd_dc_diff
                + r * L_dis_d_r * 0.1,
                "Distangle_cls_dis": L_cls_dc + L_cls_dd,
                "Distangle_cls_adv": L_cls_adv_dd,
                "Domain_adv": L_adv_d_c * a,
            }
        )

        self._update_logs(
            {
                # domain_dis
                "DomDis/d": L_dis_d,
                "DomDis/d_r": L_dis_d_r,
                "DomDis/dd": L_dis_dd,
                "DomDis/dc": L_dis_dc,
                # "DomDis/dc_c": L_dis_dc_c,
                "DomDis/d_adv_c": L_adv_d_c,
                # clses
                "Cls/c": L_cls,
                "Cls/dc": L_cls_dc,
                "Cls/dd": L_cls_dd,
                "Cls/ent": Lt['cls']['cls'],
                # recs
                # "Rec/f": L_rec_f,
                # "Rec/d": L_rec_d,
                # muts
                # "Mut/f": L_mut_f,
                "Mut/d": L_mut_d,
                # diff
                "Diff/d": L_dd_dc_diff,
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
        predcition[props < 0.7] = self.cls_info["cls_num"]
        return predcition