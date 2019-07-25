import itertools

import numpy as np
import torch

from .networks.networks import DomainClassifier

from ..basic_module import DAModule, EndlessIter
from .params import get_params


from mdata.partial.partial_dataset import require_openset_dataloader
from mdata.partial.partial_dataset import OFFICE_CLASS
from mdata.transfrom import get_transfrom

param = get_params()


def get_lambda(iter_num, max_iter, high=1.0, low=0.0, alpha=10.0):
    return np.float(
        2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter))
        - (high - low)
        + low
    )
    # return 1


def get_lr_scaler(
    iter_num, max_iter, init_lr=param.lr, alpha=10, power=0.75
):
    lr_scaler = np.float((1 + alpha * (iter_num / max_iter)) ** (-power))
    return lr_scaler


class OpensetBackprop(DAModule):
    def __init__(self):
        super().__init__(param)

        self.early_stop = self.total_steps / 2

        source_class = set(OFFICE_CLASS[0:10])
        target_class = set(OFFICE_CLASS[0:10] + OFFICE_CLASS[20:31])

        assert len(source_class.intersection(target_class)) == 10
        assert len(source_class) == 10 and len(target_class) == 21

        class_num = len(source_class) + 1

        self.class_num = class_num
        self.source_class = source_class
        self.target_class = target_class

        self.DECISION_BOUNDARY = self.TARGET.fill_(0.5)

        self._all_ready()

    def _prepare_data(self):

        back_bone = "alexnet"
        source_ld, target_ld, valid_ld = require_openset_dataloader(
            source_class=self.source_class,
            target_class=self.target_class,
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
                # reversed_coeff=lambda: get_lambda(self.current_step, self.total_steps),
                reversed_coeff=lambda: 1
            )

        return {"F": F, "C": C}

    def _regist_losses(self):

        optimer = {
            "type": torch.optim.SGD,
            "lr": param.lr,
            "momentum": 0.9,
            "weight_decay": 0.001,
        }


        self.define_loss("global_looss", networks=["C"], optimer=optimer)

        self.define_log("classify", "adv", group="train")

    def _train_step(self, s_img, s_label, t_img, t_label):

        g_source_feature = self.F(s_img)
        g_target_feature = self.F(t_img)

        class_prediction, _ = self.C(g_source_feature, adapt=False)
        _, unkonw_prediction = self.C(g_target_feature, adapt=True)

        loss_classify = self.ce(class_prediction, s_label)

        loss_adv = self.bce(unkonw_prediction, self.DECISION_BOUNDARY)

        self._update_logs({"classify": loss_classify, "adv": loss_adv})
        self._update_loss("global_looss", loss_classify + loss_adv)

        del loss_classify, loss_adv

    def _valid_step(self, img):
        feature = self.F(img)
        prediction, _ = self.C(feature)      
        return prediction