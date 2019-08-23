"""this file gives the information needed to reconstruct official Caffe models in PyTorch"""
import torch
import torch.nn as nn
from mmodel.basic_module import WeightedModule

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
            init.constant_(m.bias, 0.1)
        elif isinstance(m, nn.Module):
            weights_init_helper(m)

        if isinstance(m, WeightedModule):
            m.has_init = True

class AlexNet(WeightedModule):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=1e-04, beta=0.75, k=1),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=1e-04, beta=0.75, k=1),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),  # 0
            nn.ReLU(inplace=True),  # 1
            nn.Dropout(0.5),  # 2
            nn.Linear(4096, 4096),  # 3
            nn.ReLU(inplace=True),  # 4
            nn.Dropout(0.5),  # 5
            nn.Linear(4096, num_classes),  # 6
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model_path = "./DATASET/alexnet_caffe.pth"
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model)
    return model

class AlexFeature(WeightedModule):
    def __init__(self):
        super(AlexFeature, self).__init__()

        model_alexnet = alexnet(pretrained=True)

        self.features = model_alexnet.features

        
        self.fc = nn.Sequential()
        for i in range(6):
            self.fc.add_module(
                "classifier" + str(i), model_alexnet.classifier[i]
                )

        self.has_init = True

    def forward(self, input_data):
        feature = self.features(input_data)
        feature = feature.view(-1, 256 * 6 * 6)
        feature = self.fc(feature)       
        return feature

class AlexPto(WeightedModule):

    def __init__(self, class_num=512, bottleneck_dim=512):
        super(AlexPto, self).__init__()

        
        self.bottleneck = nn.Sequential(
            nn.Linear(4096, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            # nn.Dropout(),
            nn.LeakyReLU(),
        )

        self.fc = nn.Linear(bottleneck_dim, 256)

        weights_init_helper(self)
        self.has_init = True

    def forward(self, input_data):
        feature = self.bottleneck(input_data)
        feature = self.fc(feature)
        return feature

