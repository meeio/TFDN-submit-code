import itertools

import numpy as np
import torch

from mdata.data_iter import EndlessIter
from mdata.sampler.balanced_sampler import BalancedSampler
from mdata import for_dataset, for_digital_transforms

from torch.utils.data.dataloader import DataLoader

from ..basic_module import TrainableModule
from .params import get_params

from .networks.networks import LeNetEncoder

from mground.math_utils import euclidean_dist
from torch.nn import functional as F


# get params from defined basic params fucntion
param = get_params()



def dist_based_prediction(pred_pto, center_pto):
    dist = euclidean_dist(pred_pto, center_pto)
    return -1 * dist


class ProtopyteNetwork(TrainableModule):
    def __init__(self):
        super(ProtopyteNetwork, self).__init__(param)

        self.ce = torch.nn.CrossEntropyLoss()

        # somethin you need, can be empty
        self._all_ready()
    
    def cls_prototypes(self, ptos, targets):
        cls_num = self.data_info['cls_num']
        cls_ptos = torch.cat(
            [(ptos[targets.eq(c)]).mean(dim=0, keepdim=True) for c in range(cls_num)]
        )
        valid_cls = (cls_ptos == cls_ptos).min(dim=1)[0]
        return cls_ptos, valid_cls

    def _prepare_data(self):
        """
            prepare your dataset here
            and return a iterator dic
        """

        t = for_digital_transforms(is_rgb=False)
        train_set, t_info = for_dataset(
            "mnist", split="train", transfrom=t
        )
        valid_set, _ = for_dataset("mnist", split="valid", transfrom=t)
        data_info = {"cls_num": 10}

        sampler = BalancedSampler(t_info["targets"])
        train_loader = DataLoader(
            train_set, batch_size=128, sampler=sampler
        )
        valid_loader = DataLoader(valid_set, batch_size=128)

        iters = {
            "train": EndlessIter(train_loader),
            "valid": EndlessIter(valid_loader),
            "pre_valid": EndlessIter(train_loader, max=10)
        }

        return data_info, iters

    def _feed_data(self, mode, *args, **kwargs):

        assert mode in ["train", "pre_valid", "valid"]

        its = self.iters[mode]
        if mode == "train":
            return its.next()
        elif mode == "pre_valid":
            return its.next(need_end=True)
        else:
            return its.next(need_end=True)
        

    def _regist_networks(self):
        net = LeNetEncoder()
        return {"N": net}

    def _regist_losses(self):

        optimer = {
            "type": torch.optim.SGD,
            "lr": 0.01,
            "weight_decay": 0.001,
            # "momentum": 0.9,
            # "nesterov": True,
        }

        lr_scheduler = {
            "type": torch.optim.lr_scheduler.StepLR,
            "step_size": self.total_steps / 3,
        }

        self._define_loss(
            "claasify_loss",
            networks_key=["N"],
            optimer=optimer,
            decay_op=lr_scheduler,
        )

    def _train_process(self, datas):
        img, label = datas

        prototypes = self.N(img)

        cls_ptos, valid_idx = self.cls_prototypes(prototypes, label)
        assert (valid_idx==1).all()
        print(tttt)
        predictions = dist_based_prediction(prototypes, cls_ptos)

        loss = self.ce(predictions, label)

        self._update_loss("claasify_loss", loss)
    

    def _eval_process(self, datas):
        img, label = datas
        prototypes = self.N(img)
        predcition = dist_based_prediction(prototypes, self.valid_ptos)
        predcition = torch.max(predcition, dim=1)[1]
        return predcition

    def eval_module(self, **kwargs):
        cls_ptos = list()
        for _,i in self.networks.items():
            i.eval()
        while True:
            datas = self._feed_data_with_anpai(mode="pre_valid")
            if datas is not None:
                img, label = datas
                ptos = self.N(img)
                batch_cls_ptos = cls_prototypes(ptos, label).detach()
                cls_ptos.append(batch_cls_ptos)
            else:
                cls_ptos = torch.cat(
                    [i.unsqueeze(1) for i in cls_ptos],
                    dim = 1
                ).mean(dim=1)
                self.valid_ptos = cls_ptos
                break

        TrainableModule.eval_module(self, **kwargs)
