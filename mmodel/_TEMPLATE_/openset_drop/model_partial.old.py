import itertools

import numpy as np
import torch

from .networks.networks import DomainClassifier

from ..basic_module import DAModule, EndlessIter
from .params import get_params


from mdata.partial.partial_dataset import require_openset_dataloader
from mdata.partial.partial_dataset import OFFICE_CLASS
from mdata.transfrom import get_transfrom
import numpy as np
import torch.nn.functional as F

param = get_params()

old_ne = 0


def norm_entropy(p, reduction="None"):
    p = p.detach()
    n = p.size()[1] - 1
    p = torch.split(p, n, dim=1)[0]
    p = torch.nn.Softmax(dim=1)(p)
    e = p * torch.log((p)) / np.log(n)
    ne = -torch.sum(e, dim=1)
    if reduction == "mean":
        ne = torch.mean(ne)
    elif reduction == "top5":
        ne, _ = torch.topk(ne, 5, dim=0, largest=False)
        ne = torch.mean(ne)
    elif reduction == "top5_m":
        global old_ne
        # ne, _ = torch.topk(ne, 5, dim=0, largest=False)
        ne, _ = torch.topk(ne, 5, dim=0, largest=True)
        ne = torch.mean(ne)
        if old_ne == 0:
            old_ne = ne
        else:
            old_ne = 0.2 * old_ne + 0.8 * ne
        return old_ne
    return ne


def get_bias(iter_num, max_iter, high, low, alpha=30, center=0.1):

    p = iter_num / max_iter

    z = (
        (
            1 / (1 + np.exp(-alpha * (p - center)))
            - 1 / (1 + np.exp(-alpha * (-center)))
        )
        * ((1 + np.exp(alpha * center)) / np.exp(alpha * center))
        * (high-low)
    )

    return high - z


def get_lambda(iter_num, max_iter, high=1.0, low=0.0, alpha=10.0):

    return np.float(
        2.0
        * (high - low)
        / (1.0 + np.exp(-alpha * iter_num / max_iter))
        - (high - low)
        + low
    )


def get_lr_scaler(
    iter_num, max_iter, init_lr=param.lr, alpha=10, power=0.75
):

    if iter_num < param.task_ajust_step:
        return 1

    iter_num -= param.task_ajust_step
    max_iter -= param.task_ajust_step

    lr_scaler = np.float((1 + alpha * (iter_num / max_iter)) ** (-power))
    return lr_scaler


class PartialDrop(DAModule):
    def __init__(self):
        super().__init__(param)

        # self.eval_after = int(0.15 * self.total_steps)

        self.early_stop = self.total_steps / 2

        source_class = set(OFFICE_CLASS[0:31])
        target_class = set(OFFICE_CLASS[0:10])

        assert len(source_class.intersection(target_class)) == 10
        assert len(source_class) == 31 and len(target_class) == 10

        self.source_class = source_class
        self.target_class = target_class
        self.class_num = len(self.source_class) + 1

        self.element_bce = torch.nn.BCELoss(reduction="none")
        self.bce = torch.nn.BCELoss()
        self.element_ce = torch.nn.CrossEntropyLoss(reduction="none")
        self.DECISION_BOUNDARY = self.TARGET.fill_(1)

        self._all_ready()

    @property
    def dynamic_offset(self):
        high = 1

        return get_bias(
            self.current_step,
            self.total_steps,
            high=high,
            low = high/2,
            alpha=10,
            center=0.2,
        )

    def _prepare_data(self):

        back_bone = "alexnet"
        source_ld, target_ld, valid_ld = require_openset_dataloader(
            source_class=self.source_class,
            target_class=self.target_class,
            train_transforms=get_transfrom(back_bone, is_train=True),
            valid_transform=get_transfrom(back_bone, is_train=False),
            params=self.params,
        )

        iters = {
            "train": {
                "S": EndlessIter(source_ld),
                "T": EndlessIter(target_ld),
            },
            "valid": EndlessIter(valid_ld),
        }

        return None, iters

    def _regist_networks(self):

        if True:
            from .networks.alex import AlexNetFc, AlexGFC, AlexClassifer

            F = AlexNetFc(need_train=False)
            G = AlexGFC()
            C = AlexClassifer(
                class_num=self.class_num, 
                reversed_coeff=lambda: get_lambda(
                    self.current_step, self.total_steps
                ),
            )

        return {"F": F, "G": G, "C": C}

    def _regist_losses(self):

        optimer = {
            "type": torch.optim.SGD,
            "lr": param.lr,
            "momentum": 0.9,
            "weight_decay": 0.001,
            # "nesterov": True,
        }

        lr_scheduler = {
            "type": torch.optim.lr_scheduler.MultiStepLR,
            "gamma": 0.1,
            "milestones": [
                (
                    (self.total_step / 3)
                    + param.task_ajust_step
                    + param.pre_adapt_step
                )
            ],
        }

        self.define_loss(
            "losses",
            networks=["G", "C"],
            optimer=optimer,
            decay_op=lr_scheduler,
        )

        self.define_log("valid_loss", "valid_accu", group="valid")
        self.define_log(
            "classify", "adv", "keep", "tolorate","valid","outlier","ve","oe","base", group="train"
        )

    def example_select(self, target_entropy, base_line, mode="upper"):
        assert mode == "s"

        # in this case, target is source domain entropy
        # base_line is target domain entorpy
        # which means mean(base_line)>mean(target_entropy)

        if self.current_step > param.task_ajust_step:
            # _, top_idx = torch.topk(target_entropy, 1, dim=0, largest=True)
            # allowed_idx = target_entropy.clone().fill_(0)        
            # allowed_idx = allowed_idx.index_fill_(0, top_idx, 1)
            # ne, allowed_idx = torch.topk(allowed_idx, 5, dim=0, largest=True)
            allowed_idx = target_entropy + self.dynamic_offset > base_line
            print(base_line)
            print(target_entropy)
        else:
            # allowed_idx = (
            #     torch.abs(target_entropy - base_line) < self.dynamic_offset
            # )
            allowed_idx = target_entropy.clone().fill_(1)     
        return allowed_idx.byte()

    def _train_step(self, s_img, s_label, t_img, t_label):

        if self.current_step == self.early_stop:
            raise Exception("early stop")

        g_source_feature = self.G(self.F(s_img))
        g_target_feature = self.G(self.F(t_img))

        s_predcition, _ = self.C(g_source_feature, adapt=False)
        t_prediction, t_domain = self.C(g_target_feature, adapt=True)

        ew_loss_classify = self.element_ce(s_predcition, s_label)
        adv_loss = self.bce(t_domain, self.DECISION_BOUNDARY)

        
        ew_target_entropy = norm_entropy(t_prediction, reduction='none')
        a = F.softmax(t_prediction, dim=1) * ew_target_entropy.unsqueeze(1)
        m = torch.mean(a, dim=0)
        m = m / torch.max(m)

        source_weight = torch.gather(m, 0, s_label)
        print(source_weight)

        # source_entropy = norm_entropy(s_predcition, reduction="none")
        # base_line = norm_entropy(t_prediction, reduction="top5_m")

        # allowed_idx = self.example_select(source_entropy, base_line, "s")

        # allowed_data_label = torch.masked_select(t_label, mask=allowed_idx)
        # valid = torch.sum(allowed_data_label != self.class_num - 1)
        # outlier = torch.sum(allowed_data_label == self.class_num - 1)
        # drop = self.params.batch_size - valid - outlier

        # drop_prop = 1 - keep_prop
        # dis_loss = torch.mean(ew_dis_loss * (1 - allowed_idx)) * drop_prop
        # adv_loss = torch.mean(ew_dis_loss * allowed_idx) * keep_prop
        # allowed_data_label = torch.masked_select(s_label, mask=allowed_idx)
        # valid = torch.sum(allowed_data_label < 10)
        # outlier = torch.sum(allowed_data_label >= 10)

        # valid_idx = s_label > 10
        # outlier_idx = s_label < 10
        # valid_entropy = torch.mean(torch.masked_select(source_entropy, valid_idx))
        # outlier_entropy = torch.mean(torch.masked_select(source_entropy, outlier_idx))

        allowed_idx = allowed_idx.float().unsqueeze(1)
        keep_prop = torch.sum(allowed_idx) / self.params.batch_size
        drop_prop = 1 - keep_prop
        loss_classify = torch.mean(ew_loss_classify * allowed_idx)


        self._update_logs(
            {
                "classify": loss_classify,
                "adv": adv_loss,
                "keep": keep_prop.float() * param.batch_size,
                "valid": valid.float(),
                "outlier": outlier.float(),
                "tolorate": self.dynamic_offset,
                "ve":valid_entropy,
                "oe":outlier_entropy,
                "base":base_line,
            }
        )

        if self.current_step > param.task_ajust_step:
            self._update_losses({"losses": loss_classify})
        else:
            self._update_losses({"losses": loss_classify})

        del loss_classify, adv_loss

    def _valid_step(self, img):
        feature = self.G(self.F(img))
        prediction, _ = self.C(feature, adapt=False)
        return torch.split(prediction, self.class_num-1, dim=1)[0]
