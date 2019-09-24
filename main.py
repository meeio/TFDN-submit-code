import os
confs = [
    ["1", 8008],
    ["2", 8009],
    ["3", 8007],
]
conf = confs[0]

os.environ["CUDA_VISIBLE_DEVICES"] = conf[0]
port = conf[1]

import torch
from main_aid import TBHandler
from mmodel import get_module


if __name__ == "__main__":

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(960301)
    torch.cuda.manual_seed_all(960301)

    model_name = "DEMO1"
    tb = TBHandler(model_name)
    param, model = get_module(model_name)

    try:
        model.writer = tb.get_writer()
        tb.star_shell_tb(port)
        model.train_module()

    finally:
        tb.kill_shell_tb()
        raise

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

