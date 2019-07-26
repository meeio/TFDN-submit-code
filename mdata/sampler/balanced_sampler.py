


import random
import itertools
import numpy
import random
from torch.utils.data import Sampler
from functools import partial
from copy import deepcopy

class BalancedSampler(Sampler): 
    """ sampler for balanced sampling
    """


    def __init__(self, targets, max_per_cls=None):
        star_class = min(targets)
        end_class = max(targets)
        number_of_classes =end_class - star_class + 1
        class_wise_idxs = list()
        for idx in range(number_of_classes):
            current_class = star_class + idx
            sample_indexes = [
                i for i in range(len(targets)) if targets[i] == current_class
            ]

            if max_per_cls:
                random.shuffle(sample_indexes)
                sample_indexes = sample_indexes[0:max_per_cls]

            class_wise_idxs.append(sample_indexes)

        cls_num = len(class_wise_idxs)    

        self.cls_num = cls_num
        self.cls_idxs = class_wise_idxs
        self.total = cls_num * min([len(i) for i in class_wise_idxs])


    def __iter__(self):

        cls_idxs = deepcopy(self.cls_idxs)

        def replace_shuffle(l):
            random.shuffle(l)
            return l
        cls_idxs = list(map(replace_shuffle, cls_idxs))
        cls_idxs = list(zip(*cls_idxs))

        sample_idx = list(itertools.chain(*cls_idxs))
        return iter(sample_idx)

                    
    def __len__(self):
        return self.total
    
        


if __name__ == "__main__":
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms

    trans = transforms.Compose([  
            transforms.ToTensor(),
        ])

    minist = MNIST(
        root="./DATASET/MNIST", train=True, download=True, transform=trans
    )
    a = minist.targets.numpy()
    s = BalancedSampler(a)
    
    from torch.utils.data import DataLoader
    data = DataLoader(minist, batch_size=120, shuffle=False, sampler=s)
    # it = iter(data)

    for batch_idx, samples in enumerate(data):
        print(batch_idx)
