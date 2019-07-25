
import os
from mtrain.watcher import watcher
from mtrain.loss_driven.lossholder import LossChangeListener, LossBuket

class LogCapsule(LossChangeListener):
    def __init__(
        self,
        loss_bucker: LossBuket,
        name,
        step=1,
        to_file=False
    ):

        self.tag = name
        self.current_step = 0
     
        self.range_loss = None
        self.range_step = 0

        loss_bucker.add_lister(self)

    def avg_range_loss(self):
        try:
            result = self.range_loss / self.range_step
        except:
            result = 0.0
        self.range_loss = None
        self.range_step = 0.0
        self.range_step = 0.0
        try:
            loss = result.item()
        except:
            loss = result

        return loss


    def before_change(self):
        pass
    
    def in_change(self, value):
        try:
            value = value.clone().detach()
        except:
            value = value

        if self.range_loss is None:
            self.range_loss = value
        else:
            self.range_loss += value


    def after_change(self):
        self.current_step += 1
        self.range_step += 1


if __name__ == "__main__":
    pass

