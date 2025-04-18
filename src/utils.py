import matplotlib.pyplot as plt
import numpy as np
import torch


def dice_loss(pred, target, smooth=1e-5):
    
    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def bce_dice_loss(pred, target):
    
    bce = torch.nn.BCELoss()(pred, target)
    return bce + dice_loss(pred, target)