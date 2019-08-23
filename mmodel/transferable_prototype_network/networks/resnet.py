import torch
import torch.nn as nn
from mmodel.basic_module import WeightedModule
from torchvision.models import resnet50

def weights_init_helper(modul, params=None):
    """give a module, init it's weight
    
    Args:
        modul (nn.Module): torch module
        params (dict, optional): Defaults to None. not used.
    """
    import torch.nn.init as init

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
        elif isinstance(m, nn.BatchNorm1d):
            if m.weight is not None:
                init.normal_(m.weight, mean=1.0, std=0.02)
                init.constant_(m.bias, 0)
        # init full connect norm
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Module):
            weights_init_helper(m)

        if isinstance(m, WeightedModule):
            m.has_init = True

class ResNetFc(WeightedModule):
    def __init__(self):
        super(ResNetFc, self).__init__()
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

class ResNetPto(WeightedModule):

    def __init__(self, class_num=512, bottleneck_dim=256):
        super(ResNetPto, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(2048, bottleneck_dim),
            # nn.BatchNorm1d(bottleneck_dim),
            # nn.Dropout(),
            # nn.ReLU(),
        )

        self.fc = nn.Linear(bottleneck_dim, 128)

        weights_init_helper(self)
        self.has_init = True

    def forward(self, input_data):
        feature = self.bottleneck(input_data)
        feature = self.fc(feature)
        return feature

if __name__ == "__main__":
    m = resnet50(pretrained=True)

    print(m.fc.in_features)

