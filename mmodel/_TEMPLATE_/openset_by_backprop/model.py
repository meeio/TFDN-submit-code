import itertools

import numpy as np
import torch

from .networks.networks import DomainClassifier

from ..basic_module import DAModule, EndlessIter
from .params import get_params


from mdata.partial.partial_dataset import require_openset_dataloader
from mdata.partial.partial_dataset import OFFICE_CLASS
from mdata.transfrom import get_transfrom
from mground.gpu_utils import anpai

param = get_params()


def binary_entropy(p):
    p = p.detach()
    e = p * torch.log((p)) + (1 - p) * torch.log((1 - p))
    e = torch.mean(e) * -1
    return e


def norm_entropy(p, reduction="None", all=True):
    p = p.detach()
    n = p.size()[1] - 1
    if not all:
        p = torch.split(p, n, dim=1)[0]
    p = torch.nn.Softmax(dim=1)(p)
    e = p * torch.log((p)) / np.log(n)
    ne = -torch.sum(e, dim=1)
    if reduction == "mean":
        ne = torch.mean(ne)
    return ne


def get_lambda(iter_num, max_iter, high=1.0, low=0.0, alpha=10.0):
    return np.float(
        2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter))
        - (high - low)
        + low
    )


def get_lr_scaler(
    iter_num, max_iter, init_lr=param.lr, alpha=10, power=0.75
):
    lr_scaler = np.float((1 + alpha * (iter_num / max_iter)) ** (-power))
    return lr_scaler


class OpensetBackprop(DAModule):
    def __init__(self):
        super().__init__(param)

        ## NOTE classes setting adapt from <opensetDa by backprop>

        source_class = set(OFFICE_CLASS[0:10])
        bias_class = set(OFFICE_CLASS[10:31])
        target_class = set(OFFICE_CLASS[0:10])

        assert len(source_class.intersection(target_class)) == 10
        assert (
            len(source_class) == 10
            and len(target_class) == 10
            and len(bias_class) == 21
        )

        class_num = len(source_class) + 1 + len(bias_class)

        self.class_num = class_num
        self.source_class = source_class
        self.target_class = target_class
        self.bias_class = bias_class

        self.DECISION_BOUNDARY = self.TARGET.fill_(1)

        self._all_ready()

    def _prepare_data(self):

        back_bone = "alexnet"

        source_ld, target_ld, valid_ld = require_openset_dataloader(
            source_class=self.source_class,
            target_class=self.target_class,
            train_transforms=get_transfrom(back_bone, is_train=True),
            valid_transform=get_transfrom(back_bone, is_train=False),
            params=self.params,
        )

        bias_ld, all_ld, _ = require_openset_dataloader(
            source_class=self.bias_class,
            target_class=self.bias_class ,
            train_transforms=get_transfrom(back_bone, is_train=True),
            valid_transform=get_transfrom(back_bone, is_train=False),
            params=self.params,
            class_wiese_valid=param.classwise_valid,
        )

        iters = {
            "train": {
                "S": EndlessIter(source_ld),
                "T": EndlessIter(target_ld),
            }
        }

        if param.classwise_valid:
            iters['valid'] = {k: EndlessIter(v) for k,v in valid_ld.items()}
            
        else:
            iters['valid'] = EndlessIter(valid_ld)
                
        return None, iters

    def _regist_networks(self):

        if True:
            from .networks.alex import AlexNetFc, AlexClassifer

            F = AlexNetFc()
            C = AlexClassifer(
                class_num=self.class_num,
                # reversed_coeff=lambda: get_lambda(
                #     self.current_step, self.total_steps
                # ),
                reversed_coeff=lambda: 1
            )

        return {"F": F, "C": C}

    def _regist_losses(self):

        optimer = {
            "type": torch.optim.SGD,
            "lr": param.lr,
            "momentum": 0.9,
            "weight_decay": 0.001,
            # "nesterov": True,
            # "lr_mult": {"F": 0.1},
        }

        self.define_loss("global_looss", networks=["C"], optimer=optimer)

        self.define_log("valid_loss", "valid_accu", group="valid")
        self.define_log("classify", "adv", "se", "be", "te", group="train")

    def _train_step(self, s_img, s_label, t_img, t_label):

        b_img, b_label = self.iters["train"]["B"].next()
        b_img, b_label = anpai((b_img, b_label), True, False)

        source_f = self.F(s_img)
        bias_f = self.F(b_img)
        target_f = self.F(t_img)

        s_cls_p, s_un_p = self.C(source_f, adapt=False)
        b_cls_p, b_un_p = self.C(bias_f, adapt=False)
        t_cls_p, t_un_p = self.C(target_f, adapt=True)

        s_loss_classify = self.ce(s_cls_p, s_label)
        b_loss_classify = self.ce(b_cls_p, b_label)

        loss_adv = self.bce(t_un_p, self.DECISION_BOUNDARY)

        if self.current_step < 200:
            loss_classify = 0.2 * s_loss_classify + 0.8 * b_loss_classify

        else:
            loss_classify = s_loss_classify
        
        if self.current_step < 100:
            self._update_loss("global_looss", loss_classify)

        else:
            self._update_loss("global_looss", loss_classify + loss_adv)


        self._update_logs(
            {
                "classify": loss_classify,
                "adv": loss_adv,
                "se": norm_entropy(s_cls_p, reduction="mean"),
                "be": norm_entropy(b_cls_p, reduction="mean"),
                "te": norm_entropy(t_cls_p, reduction="mean"),
            }
        )

        del loss_classify, loss_adv

    def _valid_step(self, img):
        feature = self.F(img)
        prediction, _ = self.C(feature)
        return torch.split(prediction, self.class_num - 1, dim=1)[0]
