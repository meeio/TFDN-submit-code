import itertools

import numpy as np
import torch

from mdata.data_iter import EndlessIter
from mdata.sampler.balanced_sampler import BalancedSampler
from mdata import for_dataset, for_digital_transforms

from torch.utils.data.dataloader import DataLoader

from ..basic_module import TrainableModule
from .tpn_params import get_params

from .networks.networks import LeNetEncoder

from mground.math_utils import euclidean_dist
from mground.loss_utils import mmd_loss
from torch.nn import functional as F

import math


# get params from defined basic params fucntion
param = get_params()


def dist_based_prediction(pred_pto, center_pto):
    dist = euclidean_dist(pred_pto, center_pto)
    dist = -1 * dist
    pred = torch.nn.functional.log_softmax(dist, dim=1)
    return pred


class TransferableProtopyteNetwork(TrainableModule):
    def __init__(self):
        super().__init__(param)

        self.nll = torch.nn.NLLLoss()
        self.log_softmax = torch.nn.LogSoftmax()
        self.mmd = mmd_loss
        self.pseudo_threshold = math.log(0.6)

        # somethin you need, can be empty
        self._all_ready()

    def cls_ptos(self, ptos, targets):
        cls_num = self.data_info["cls_num"]
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

        trans = for_digital_transforms(is_rgb=False)
        sou_set, s_info = for_dataset(
            "mnist", split="train", transfrom=trans
        )
        tar_set, _ = for_dataset("usps", split="train", transfrom=trans)
        val_set, _ = for_dataset("usps", split="test", transfrom=trans)

        sou_loader = DataLoader(
            sou_set,
            batch_size=128,
            drop_last=True,
            sampler=BalancedSampler(s_info["labels"], max_per_cls=200),
        )
        tar_loader = DataLoader(
            tar_set, batch_size=128, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(val_set, batch_size=128, drop_last=True)

        iters = {
            "train": {
                "sou_iter": EndlessIter(sou_loader),
                "tar_iter": EndlessIter(tar_loader),
            },
            "valid": {
                "sou_iter": EndlessIter(sou_loader, max=10),
                "val_iter": EndlessIter(val_loader),
            },
        }

        data_info = {"cls_num": len(torch.unique(s_info["labels"]))}
        return data_info, iters

    def _feed_data(self, mode, *args, **kwargs):
        assert mode in ["train", "pre_valid", "valid"]

        if mode == "train":
            its = self.iters["train"]
            return its["sou_iter"].next() + its["tar_iter"].next()
        elif mode == "pre_valid":
            its = self.iters["valid"]
            return its["sou_iter"].next(need_end=True)
        elif mode == "valid":
            its = self.iters["valid"]
            return its["val_iter"].next(need_end=True)

        raise Exception("feed error!")

    def _regist_networks(self):
        net = LeNetEncoder()
        return {"N": net}

    def _regist_losses(self):

        optimer = {
            "type": torch.optim.Adam,
            "lr": 0.0002,
            "weight_decay": 0.0005,
            # "nesterov": True,
            # "momentum": 0.95,
            "betas": (0.9,0.999),
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

        self._define_log("cls_loss","gen_loss", group="train")



    def _train_process(self, datas):

        s_imgs, s_labels, t_imgs, _ = datas

        # generate sample prototypes and center prototypes
        s_ptos = self.N(s_imgs)
        t_ptos = self.N(t_imgs)
        s_pto_cets, s_pto_idxs = self.cls_ptos(s_ptos, s_labels)

        # make prediction based on prototypes
        s_preds = dist_based_prediction(s_ptos, s_pto_cets)
        t_preds = dist_based_prediction(t_ptos, s_pto_cets)

        # generate pseudo labels and targets prototypes 
        t_preds_prop, t_preds_labels = torch.max(t_preds, dim=1)
        t_pseu_idx = t_preds_prop > self.pseudo_threshold
        t_ptos = t_ptos[t_pseu_idx]
        t_pseu_labels = t_preds_labels[t_pseu_idx]
        _, t_pto_idxs = self.cls_ptos(t_ptos, t_pseu_labels)

        # generate source-targets prototypes
        st_ptos = torch.cat([s_ptos, t_ptos], dim=0)
        st_labels = torch.cat([s_labels, t_pseu_labels], dim=0)
        _, st_pto_idx = self.cls_ptos(st_ptos, st_labels)

        # calculate general loss
        _L_gen = list()
        for (s,t,st) in zip(s_pto_idxs, t_pto_idxs, st_pto_idx):
            if t.sum() > 1:                
                _L_gen.append(self.mmd(s_ptos[s], t_ptos[t]))
                _L_gen.append(self.mmd(s_ptos[s], st_ptos[st]))
                _L_gen.append(self.mmd(t_ptos[t], st_ptos[st]))
        
        L_gen = sum(_L_gen) / max(len(_L_gen), 1)

        L_cls = self.nll(s_preds, s_labels)

        L = L_gen + L_cls


        self._update_losses({
            "total": L,
            "cls_loss": L_cls,
            "gen_loss": L_gen,
        })


    def _eval_process(self, datas):
        img, label = datas
        prototypes = self.N(img)
        predcition = dist_based_prediction(prototypes, self.valid_ptos)
        predcition = torch.max(predcition, dim=1)[1]
        return predcition

    def eval_module(self, **kwargs):
        cls_ptos = list()
        for _, i in self.networks.items():
            i.eval()
        while True:
            datas = self._feed_data_with_anpai(mode="pre_valid")
            if datas is not None:           
                img, label = datas
                ptos = self.N(img)
                batch_cls_ptos,_ = self.cls_ptos(ptos, label)
                batch_cls_ptos = batch_cls_ptos.detach()
                cls_ptos.append(batch_cls_ptos)
            else:
                cls_ptos = torch.cat(
                    [i.unsqueeze(1) for i in cls_ptos], dim=1
                ).mean(dim=1)
                self.valid_ptos = cls_ptos
                break

        TrainableModule.eval_module(self, **kwargs)
