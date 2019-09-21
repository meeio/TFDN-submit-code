import torch
import torch.nn.functional as F

def ent(output):
    p = F.softmax(output, dim=-1)
    ee = -1 * p * torch.log(p+1e-6)
    me = torch.mean(ee)
    return me

def ent_v2(self, output):
    return - torch.mean(torch.log(F.softmax(output + 1e-6)))


