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
from .tpn_params import get_params

from mground.math_utils import euclidean_dist
from mground.loss_utils import mmd_loss
from torch.nn import functional as F

from functools import partial
import math

from torch.nn.functional import cosine_similarity


# get params from defined basic params fucntion
param = get_params()


def dist_based_prediction(pred_pto, center_pto):
    dist = euclidean_dist(pred_pto, center_pto, x_wise=True)
    dist = -1 * dist
    pred = torch.nn.functional.log_softmax(dist, dim=1)
    return pred

def get_lambda(iter_num, max_iter, high=1.0, low=0.0, alpha=10.0):
    return np.float(
        2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter))
        - (high - low)
        + low
    )

class TransferableProtopyteNetwork(TrainableModule):
    def __init__(self):

        clses = list(range(31))
        sou_cls = clses[0:31]
        tar_cls = clses[0:31]

        self.cet_ptos = None

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

        self.nll = torch.nn.NLLLoss()
        self.log_softmax = torch.nn.LogSoftmax()
        self.mmd = mmd_loss
        self.pseudo_threshold = math.log(0.8)
        self.unkown_threshold = math.log(0.6)

        self.max_p = torch.Tensor([0.95]).cuda()
        super().__init__(param)

    def update_cet_ptos(self, ptos):
        if self.cet_ptos is None:
            self.cet_ptos = ptos
            avg_p = 0
        else:
            old_ptos = self.cet_ptos.detach()
            p = cosine_similarity(ptos, old_ptos)
            p = (p ** 2).unsqueeze(1).detach()
            p = 0.8
            # p = 1
            self.cet_ptos = (1 - p) * old_ptos + p * ptos
            # avg_p = p.sum()/31
            avg_p = p
        return self.cet_ptos, avg_p

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
            drop_last=True,
            num_workers=0,
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
            return its["sou_iter"].next() + its["tar_iter"].next()
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
        from .networks.alex import AlexFeature, AlexPto
        return {"F": AlexFeature(), "G": AlexPto()}

    def _regist_losses(self):

        optimer = {
            "type": torch.optim.SGD,
            "lr": 0.001,
            "weight_decay": 0.0005,
            "momentum": 0.95,
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
            networks_key=["F", "G"],
            optimer=optimer,
            # decay_op=lr_scheduler,
        )

        self.mmd_lambda = partial(get_lambda, max_iter=self.total_steps)
        
        self._define_log("cls_loss", "f", group="train")

    def _train_process(self, datas):

        s_imgs, s_labels, t_imgs, t_labels = datas

        # generate sample prototypes and center prototypes
        s_ptos = self.G(self.F(s_imgs))
        # t_ptos = self.G(self.F(t_imgs))
        s_labels, cet_label = torch.split(s_labels, 64, dim=0)
        s_ptos, cet_ptos = torch.split(s_ptos, 64, dim=0)
        s_pto_cets, s_pto_idxs = self.cls_ptos(cet_ptos, cet_label)

        # # make prediction based on prototypes
        s_preds = dist_based_prediction(s_ptos, s_pto_cets)
        # t_preds = dist_based_prediction(t_ptos, s_pto_cets)
        f = torch.mm(s_pto_cets.t(), s_pto_cets) - 1
        f = torch.norm(f, p='fro') ** 2
        f = f * 0.00005

        # # generate pseudo labels and targets prototypes
        # t_preds_prop, t_preds_labels = torch.max(t_preds, dim=1)
        # t_pseu_idx = t_preds_prop > self.pseudo_threshold
        # t_ptos = t_ptos[t_pseu_idx]
        # t_pseu_labels = t_preds_labels[t_pseu_idx]
        # _, t_pto_idxs = self.cls_ptos(t_ptos, t_pseu_labels)

        # # generate source-targets prototypes
        # st_ptos = torch.cat([s_ptos, t_ptos], dim=0)
        # st_labels = torch.cat([s_labels, t_pseu_labels], dim=0)
        # _, st_pto_idx = self.cls_ptos(st_ptos, st_labels)

        # # calculate general loss
        # _L_gen = list()
        # for (s, t, st) in zip(s_pto_idxs, t_pto_idxs, st_pto_idx):
        #     if t.sum() > 1:
        #         _L_gen.append(self.mmd(s_ptos[s], t_ptos[t]))
        #         _L_gen.append(self.mmd(s_ptos[s], st_ptos[st]))
        #         _L_gen.append(self.mmd(t_ptos[t], st_ptos[st]))

        # L_gen = sum(_L_gen) / max(len(_L_gen), 1)
        # L_gen = L_gen / 3

        L_cls = self.nll(s_preds, s_labels)

        L =  L_cls

        # print(L)

        self._update_losses(
            {
                "total": L,
                "cls_loss": L_cls,
                "f": f,
                # "gen_loss": L_gen,
                # "p": p,
            }
        )

    def _eval_process(self, datas):
        img, label = datas
        prototypes = self.G(self.F(img))
        predcition = dist_based_prediction(prototypes, self.valid_ptos)
        props, predcition = torch.max(predcition, dim=1)
        # predcition[props < self.unkown_threshold] = self.cls_info["unkown"]
        return predcition

    def eval_module(self, **kwargs):
        cls_ptos = list()
        for _, i in self.networks.items():
            i.eval()
        while True:
            datas = self._feed_data_with_anpai(mode="pre_valid")
            if datas is not None:
                img, label = datas
                ptos = self.G(self.F(img))
                batch_cls_ptos, _ = self.cls_ptos(ptos, label)
                batch_cls_ptos = batch_cls_ptos.detach()
                cls_ptos.append(batch_cls_ptos)
            else:
                cls_ptos = torch.cat(
                    [i.unsqueeze(1) for i in cls_ptos], dim=1
                ).mean(dim=1)
                self.valid_ptos = cls_ptos
                break

        TrainableModule.eval_module(self, **kwargs)
