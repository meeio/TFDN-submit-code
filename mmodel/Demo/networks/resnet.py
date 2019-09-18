import torch
import torch.nn as nn
from mmodel.basic_module import WeightedModule
from torchvision.models import resnet50
from mmodel.utils.gradient_reverse_layer import GradReverseLayer


# def weights_init_helper(modul, params=None):
#     """give a module, init it's weight

#     Args:
#         modul (nn.Module): torch module
#         params (dict, optional): Defaults to None. not used.
#     """
#     import torch.nn.init as init

#     for m in modul.children():
#         # init Conv2d with norm
#         if isinstance(m, nn.Conv2d):
#             init.kaiming_uniform_(m.weight)
#             init.constant_(m.bias, 0)
#         # init BatchNorm with norm and constant
#         elif isinstance(m, nn.BatchNorm2d):
#             if m.weight is not None:
#                 init.normal_(m.weight, mean=1.0, std=0.02)
#                 init.constant_(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm1d):
#             if m.weight is not None:
#                 init.normal_(m.weight, mean=1.0, std=0.02)
#                 init.constant_(m.bias, 0)
#         # init full connect norm
#         elif isinstance(m, nn.Linear):
#             init.kaiming_uniform_(m.weight)
#             init.constant_(m.bias, 0)
#         elif isinstance(m, nn.Module):
#             weights_init_helper(m)

#         if isinstance(m, WeightedModule):
#             m.has_init = True


class ResnetFeat(nn.Module):
    def __init__(self):
        super().__init__()
        model_resnet = resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool,
        )

        self.has_init = True

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        return x

    def train(self, mode=True):
        # freeze BN mean and std
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d):
                module.train(False)
            else:
                module.train(mode)


class ResnetCls(nn.Module):
    """
    a two-layer MLP for classification
    """

    def __init__(self, in_dim=2048, out_dim=31, bottle_neck_dim=256):
        super(ResnetCls, self).__init__()
        self.pto = nn.Linear(in_dim, bottle_neck_dim)
        self.cls = nn.Linear(bottle_neck_dim, out_dim)

    def forward(self, x):
        ptos = self.pto(x)
        preds = self.cls(ptos)
        
        return ptos, preds


class DisNet(nn.Module):
    def __init__(self, in_feat_dim, adv_coeff_fn=lambda: 1):
        super().__init__()

        self.predictor = nn.Sequential(
            GradReverseLayer(coeff_fn=adv_coeff_fn),

            nn.Linear(in_feat_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        result = self.predictor(x)
        return result


if __name__ == "__main__":
    m = resnet50(pretrained=True)
    print(m.fc.in_features)
