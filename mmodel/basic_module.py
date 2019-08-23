import torch
import torch.nn as nn
import torch.nn.init as init

from mground.gpu_utils import anpai

import os

from abc import ABC, abstractclassmethod

from mtrain.loss_driven.lossholder import LossHolder
from mtrain.loss_driven.loger import LogCapsule
from mtrain.loss_driven.trainer import TrainCapsule

from mtrain.recorder.info_print import tabulate_print_losses
from mtrain.recorder.info_print import cprint, HINTS, WARMN, VALID, TRAIN, BUILD

from mdata.data_iter import EndlessIter


def _basic_weights_init_helper(modul, params=None):
    """give a module, init it's weight
    
    Args:
        modul (nn.Module): torch module
        params (dict, optional): Defaults to None. not used.
    """

    for m in modul.children():
        # init Conv2d with norm
        if isinstance(m, nn.Conv2d):
            init.kaiming_uniform_(m.weight)
            init.constant_(m.bias, 0)
        # init BatchNorm with norm and constant
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                init.normal_(m.weight, mean=1.0, std=0.02)
                init.constant_(m.bias, 0)
        # init full connect norm
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

        elif isinstance(m, nn.Module):
            _basic_weights_init_helper(m)

        if isinstance(m, WeightedModule):
            m.has_init = True


class WeightedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.has_init = False
        self.tag = None

    def weight_init(self, handler=None, record_path=None):
        """initial weights with help function `handler` or a path of check point `record_path` to restore, if neither of those are provide, leave init 
        jod to torch inner function.

            handler (function, optional): Defaults to None. use to init weights, such function must has sigture of `handler(Module)`.
            record_path (str, optional): Defaults to None. use to locate cheack point file.
        """

        """ init weights with torch inner function 
        """

        name = self.__class__.__name__
        str = name + "'s weights init from %s."

        infos = str % "Pytorch's inner mechainist"
        if self.has_init:
            infos = str % "class inner define"

        elif record_path is not None:
            f = torch.load(record_path)
            self.load_state_dict(f)
            infos = str % "check point"

        elif handler is not None:
            handler(self)
            infos = str % "provided init function"

        cprint(BUILD, infos)

        self.has_init = True

    def register_weight_handler(handler):
        WeightedModule.static_weight_handler = handler

    def save_model(self, path, tag):
        f = path + tag
        torch.save(self.cpu().state_dict(), f)
        return os.path.abspath(f)


class TrainableModule(ABC):

    """ An ABC, a Tranable Module is a teample class need to define
    data process, train step and eval_step

    """

    def __init__(self, params):

        super(TrainableModule, self).__init__()

        self.params = params

        # all key point of training steps
        self.log_step = self.params.log_per_step
        self.eval_step = self.params.eval_per_step
        self.eval_after_step = -1
        self.total_steps = self.params.steps
        self.current_step = 0.0
        self.current_epoch = 0.0
        self.need_pre_eval = False
        self.has_pre_eval = False
        self.class_wise_eval = True
        self.best_accurace = 0.0

        # loss changing driven
        self.loss_holder = LossHolder()
        self.train_caps = dict()
        self.train_loggers = dict()
        self.valid_loggers = dict()

        # get all networks and init weights
        networks = self._regist_networks()
        assert type(networks) is dict

        def init_weight_and_key(n, k):
            n.weight_init()
            n.tag = k

        for k, i in networks.items():
            if type(i) is nn.Sequential:
                i.tag = k
                for c in i.children():
                    init_weight_and_key(c, k)
            else:
                init_weight_and_key(i, k)

        # send networks to gup
        networks = {
            i: anpai(j, use_gpu=self.params.use_gpu)
            for i, j in networks.items()
        }

        # make network to be class attrs
        for i, j in networks.items():
            self.__setattr__(i, j)
        self.networks = networks

        # generate train dataloaders and valid dataloaders
        # data_info is a dict contains basic data infomations
        cls_num, iters = self._prepare_data()
        confusion_matrix = torch.zeros(cls_num, cls_num)

        self.confusion_matrix = confusion_matrix
        self.iters = iters
        self._define_log(*["cls_{}".format(i) for i in range(cls_num)], group="valid")
        self._define_log("valid_accurace", group="valid")

        # regist losses
        self._regist_losses()

    def _all_ready(self):
        raise Exception("Not Use Anymore")

    @abstractclassmethod
    def _prepare_data(self):
        """ handle dataset to produce dataloader
        
        Returns:
            list -- a dict of datainfo, a list of train dataloader
            and a list of valid dataloader.
        """

        data_info = dict()
        train_loaders = list()
        valid_loaders = list()
        return data_info, train_loaders, valid_loaders

    @abstractclassmethod
    def _feed_data(self, mode):
        """ feed example based on dataloaders

        Returns:
            list -- all datas needed.
        """
        datas = list()
        return datas

    @abstractclassmethod
    def _regist_losses(self):
        """ regist lossed with the help of regist loss

        Returns:
            list -- all datas needed.
        """
        return

    @abstractclassmethod
    def _regist_networks(self):
        """ feed example based on dataloaders

        Returns:
            list -- all datas needed.
        """
        networks = dict()
        return networks

    @abstractclassmethod
    def _train_process(self, datas, **kwargs):
        """process to train 
        """
        pass

    @abstractclassmethod
    def _eval_process(self, datas, **kwargs):
        """process to eval 
        """
        end_epoch = datas is None

        pass

    def _feed_data_with_anpai(self, mode):
        data = self._feed_data(mode)
        if data is not None:
            data = anpai(data, self.params.use_gpu, need_logging=False)
        return data

    def train_module(self, **kwargs):

        for _ in range(self.total_steps):

            # set all networks to train mode
            for _, i in self.networks.items():
                i.train(True)

            datas = self._feed_data_with_anpai(mode="train")
            self._train_process(datas, **kwargs)

            # making log
            if self.current_step % self.log_step == (self.log_step - 1):

                losses = [
                    (k, v.avg_range_loss())
                    for k, v in self.train_loggers.items()
                ]

                if not self.params.disable_std:

                    cprint(
                        HINTS,
                        self.params.tag
                        + " - "
                        + "Steps %3d ends. Remain %3d steps to go. Fished %.2f%%"
                        % (
                            self.current_step + 1,
                            self.params.steps - self.current_step - 1,
                            (self.current_step + 1)
                            / (self.params.steps + 1)
                            * 100,
                        ),
                    )

                    cprint(
                        HINTS,
                        "Current best accurace is %3.3f%%."
                        % (self.best_accurace * 100),
                    )

                    tabulate_print_losses(
                        losses, trace="dalosses", mode="train"
                    )

            # begain eval
            if (
                self.current_step % self.eval_step == (self.eval_step - 1)
                and self.current_step > self.eval_after_step
            ):
                self.eval_module(**kwargs)
                # set all networks to train mode
                for _, i in self.networks.items():
                    i.train(True)

            self.current_step += 1

    def eval_module(self, **kwargs):

        # set all networks to eval mode
        for _, i in self.networks.items():
            i.eval()

        # reset confusion matrix
        self.confusion_matrix.fill_(0)

        while True:
            datas = self._feed_data_with_anpai(mode="valid")
            if datas is None:
                break
            prediction = self._eval_process(datas)
            targets = datas[-1]
            for t, p in zip(targets.view(-1), prediction.view(-1)):
                self.confusion_matrix[t.long(), p.long()] += 1

        accurace = None
        cls_wise_accu = None
        if self.class_wise_eval:
            cls_wise_accu = self.confusion_matrix.diag() / self.confusion_matrix.sum(
                1
            )
            _cls_wise_accu = cls_wise_accu[cls_wise_accu == cls_wise_accu]
            accurace = torch.mean(_cls_wise_accu)

        
        self._update_losses(
            {"cls_{}".format(i): v for i, v in enumerate(cls_wise_accu)}
        )
        self._update_loss("valid_accurace", accurace)
        self.best_accurace = max((self.best_accurace, accurace))

        losses = [
            (k, v.avg_range_loss()) for k, v in self.valid_loggers.items()
        ]

        cprint(VALID, "End a evaling step.")
        tabulate_print_losses(losses, trace="validloss", mode="valid")

    def _define_loss(
        self, loss_name, networks_key, optimer: dict, decay_op: dict = None
    ):
        """registe loss according loss_name and relative networks.
        after this process the loss will bind to the weight of networks, which means this loss will used to update weights of provied networks.
        
        Arguments:
            loss_name {str} -- loss name
            networks_key {list} -- relative network names
        """

        networks_key = (
            [networks_key]
            if not isinstance(networks_key, (tuple, list))
            else networks_key
        )

        networks = [self.networks[i] for i in networks_key]

        optimer = optimer.copy()
        optimer_info = [optimer.pop("type"), optimer]

        decay_info = None
        if decay_op is not None:
            decay_op = decay_op.copy()
            decay_info = [decay_op.pop("type"), decay_op]

        loss_buck = self.loss_holder.new_loss(loss_name)
        t = TrainCapsule(
            loss_buck,
            networks,
            optimer_info=optimer_info,
            decay_info=decay_info,
        )

        self._define_log(loss_name, group="train")

        # self.train_caps[loss_name] = t

    def _define_log(self, *loss_name, group="train"):
        def log_name(name):
            loss_buck = self.loss_holder.get_loss(name, need_create=True)
            l = LogCapsule(loss_buck, name, to_file=self.params.make_record)
            if group == "train":
                self.train_loggers[name] = l
            else:
                self.valid_loggers[name] = l

        for name in loss_name:
            log_name(name)

    def _update_loss(self, loss_name, value):
        self.loss_holder.update_loss(loss_name, value)

    def _update_losses(self, a: dict):
        for index, key in enumerate(a):
            reach_last = index == (len(a) - 1)
            self._update_loss(key, a[key])


class DAModule(TrainableModule):
    def __init__(self, params):
        super(DAModule, self).__init__(params)

        self.best_accurace = 0.0
        self.total = self.corret = 0

        # set usefully loss function
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()

        self.TrainCpasule = TrainCapsule

        # # define valid losses
        # self.define_log("valid_accu", group="valid")

        # set total train step
        self.total_step = self.params.steps

        # init global label
        self.TARGET, self.SOURCE = self.__batch_domain_label(
            self.params.batch_size
        )

    @abstractclassmethod
    def _train_step(self, s_img, s_label, t_img, t_label):
        pass

    @abstractclassmethod
    def _valid_step(self, img):
        pass

    def _prepare_data(self):

        params = self.params

        dataset = params.dataset
        source = params.source
        target = params.target

        def get_set(dataset, domain, split):
            if dataset is None or dataset == "NONE":
                dataset = mdl.get_dataset(
                    dataset=domain, domain=None, split=split
                )
            else:
                dataset = mdl.get_dataset(
                    dataset=dataset, domain=domain, split=split
                )
            return dataset

        train_S_set = mdl.get_dataset(dataset, source, split="train")
        train_T_set = mdl.get_dataset(dataset, target, split="train")
        valid_set = mdl.get_dataset(dataset, target, split="test")

        def get_loader(dataset, shuffle, drop_last, batch_size=None):
            batch_size = params.batch_size if batch_size is None else batch_size
            l = torch.utils.data.DataLoader(
                dataset,
                batch_size=params.batch_size,
                drop_last=drop_last,
                shuffle=shuffle,
            )
            return l

        train_S_l = get_loader(train_S_set, shuffle=True, drop_last=True)
        train_T_l = get_loader(train_T_set, shuffle=True, drop_last=True)
        valid_l = get_loader(
            valid_set,
            shuffle=True,
            drop_last=True,
            batch_size=params.batch_size / 2,
        )

        iters = {
            "train": {"S": EndlessIter(train_S_l), "T": EndlessIter(train_T_l)},
            "valid": EndlessIter(valid_l),
        }

        return None, iters

    def _feed_data(self, mode, *args, **kwargs):

        assert mode in ["train", "valid"]

        its = self.iters[mode]
        if mode == "train":
            s_img, s_label = its["S"].next()
            t_img, t_label = its["T"].next()
            return s_img, s_label, t_img, t_label
        else:
            return its.next(need_end=True)

    def _train_process(self, datas):

        s_img, s_label, t_img, t_label = datas

        # begain a train step
        self._train_step(s_img, s_label, t_img, t_label)

    def _log_process(self):

        return losses

    def _eval_process(self, datas, **kwargs):

        params = self.params

        valid_target = kwargs.get("valid_target", "accu")

        end_epoch = datas is None

        def handle_datas(datas):

            img, label = datas

            # get result from a valid_step
            predict = self._valid_step(img)

            # calculate valid accurace and make record
            current_size = label.size()[0]

            # pred_cls = predict.data.max(1)[1]
            # corrent_count = pred_cls.eq(label.data).sum()

            _, predic_class = torch.max(predict, 1)

            corrent_count = (torch.squeeze(predic_class) == label).sum().float()

            self._update_logs(
                {"valid_" + valid_target: corrent_count * 100 / current_size},
                group="valid",
            )

            return corrent_count, current_size

        if not end_epoch:
            right, size = handle_datas(datas)
            self.total += size
            self.corret += right
        else:
            logger.log(VALID, "End a evaling {}.".format(valid_target))
            accu = self.corret / (self.total)
            self.best_accurace = max((self.best_accurace, accu))
            self.total = 0
            self.corret = 0

    def __batch_domain_label(self, batch_size):
        # Generate all Source and Domain label.
        SOURCE = 1
        TARGET = 0

        sd = torch.Tensor(batch_size, 1).fill_(SOURCE)
        td = torch.Tensor(batch_size, 1).fill_(TARGET)

        sd, td, = anpai((sd, td), self.params.use_gpu, False)
        return sd, td
