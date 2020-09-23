import torch.nn.functional as F
#import torch.nn as nn
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)

def cel_loss(output, target):
    return F.cross_entropy(output, target)
