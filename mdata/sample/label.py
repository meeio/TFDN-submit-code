import torch

class OpensetLabelManager(object):
    def __init__(self, cls_num, close_part, open_part):
        assert len(set(close_part+open_part)) == cls_num
        close_part = set(close_part)
        open_part = set(open_part)
        assert close_part != open_part
        self.unknown_labels = open_part - close_part
        self.cls_num = len(open_part) + 1
        self.unknown_cls = self.cls_num-1
        self.open_part = sorted(list(open_part))
        self.close_part = sorted(list(close_part))
    
    def conver_target_label(self, label_tensor):
        conver_idx = torch.zeros_like(label_tensor)
        for i in self.unknown_labels:
            conver_idx.masked_fill_(label_tensor == i, 1)
        label_tensor.masked_fill_(conver_idx.byte(), self.unknown_cls)

    

if __name__ == "__main__":
    l = OpensetLabelManager(10, [0,1,2,3,4,5,6], [3,4,5,6,7,8,9])

    import torch
