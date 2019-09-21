import torch
import torch.nn as nn
import torch.nn.functional as F
from mmodel.utils.gradient_reverse_layer import GradReverseLayer

from mmodel.utils.backbone import ResnetFeat

# BackBone = ResnetFeat

class Disentangler(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
    
    def forward(self, feats):
        dist_feats = self.net(feats)
        return dist_feats


class DomainDis(nn.Module):
    def __init__(self, adv_coeff_fn=lambda:-1):
        super().__init__()
        self.grl = GradReverseLayer(adv_coeff_fn)
        self.D = nn.Sequential(
            nn.Linear(2048, 256),
            nn.LeakyReLU(inplace=True),

            nn.Linear(256, 1),
            # nn.Sigmoid(),
            # nn.LeakyReLU(inplace=True),            
        )

    def forward(self, feats, adv=False):
        if adv:
            feats = self.grl(feats)            
        domain = self.D(feats)
        return domain

class ClassPredictor(nn.Module):
    def __init__(self, cls_num, adv_coeff_fn=lambda:-1):
        super().__init__()
        self.grl = GradReverseLayer(adv_coeff_fn)
        self.C = nn.Sequential(
            nn.Linear(2048, cls_num),
        )

    def forward(self, feats, adv=False):
        if adv:
            feats = self.grl(feats)   
        cls = self.C(feats)
        return cls

class Reconstructor(nn.Module):
    def __init__(self):
        super().__init__()
        self.R = nn.Sequential(
            nn.Linear(2048*2, 2048),
        )

    def forward(self, feats_1, feats_2):
        ori_feats = torch.cat([feats_1,feats_2], dim=1)
        rec_feats = self.R(ori_feats)
        return rec_feats

class Mine(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_x = nn.Linear(2048, 512)
        self.fc1_y = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512,1)
    def forward(self, x,y):
        h1 = F.leaky_relu(self.fc1_x(x)+self.fc1_y(y))
        h2 = self.fc2(h1)
        return h2