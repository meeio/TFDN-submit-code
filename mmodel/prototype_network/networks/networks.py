import torch
from torch import nn
from mmodel.basic_module import WeightedModule


def init_weights(m):
    classname = m.__class__.__name__
    if (
        classname.find("Conv2d") != -1
        or classname.find("ConvTranspose2d") != -1
    ):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class LeNetEncoder(WeightedModule):
    """LeNet encoder model."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),

            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(500, 10)
        )

    def forward(self, inputs):
        """Forward the LeNet."""
        batch_size = inputs.size()[0]
        conv_out = self.encoder(inputs)
        fc_out = self.fc(conv_out.view(batch_size, -1))
        return fc_out

