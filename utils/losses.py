import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.CE_loss = nn.CrossEntropyLoss(
            reduction=reduction, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


class BCELoss(nn.Module):
    def __init__(self, reduction="mean", pos_weight=1.0):
        pos_weight = torch.tensor(pos_weight)
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction=reduction, pos_weight=pos_weight)

    def forward(self, prediction, targets):
        return self.bce_loss(prediction, targets)


class CELoss(nn.Module):
    def __init__(self, weight=[1, 1], ignore_index=-100, reduction='mean'):
        super(CELoss, self).__init__()
        weight = torch.tensor(weight)
        self.CE = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target.squeeze(1).long())
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = torch.sigmoid(prediction)
        intersection = 2 * torch.sum(prediction * target) + self.smooth
        union = torch.sum(prediction) + torch.sum(target) + self.smooth
        loss = 1 - intersection / union
        return loss

import torch.nn.functional as F
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        targets = targets.float()
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class CE_DiceLoss(nn.Module):
    def __init__(self, reduction="mean", D_weight=0.2):
        super(CE_DiceLoss, self).__init__()
        self.DiceLoss = DiceLoss()
        self.BCELoss = BCELoss(reduction=reduction)
        self.D_weight = D_weight

    def forward(self, prediction, targets):
        return self.D_weight * self.DiceLoss(prediction, targets) + (1 - self.D_weight) * self.BCELoss(prediction,
                                                                                                       targets)


import torch
import torch.nn as nn
import kornia as K

class BCE_Connect_Loss(nn.Module):
    def __init__(self, bce_weight=0.5, con_weight=0.5):
        super(BCE_Connect_Loss, self).__init__()
        self.bce_weight = bce_weight
        self.con_weight = con_weight
        self.BCELoss = nn.BCELoss()
        self.Connect_Loss = Connect_Loss()

    def forward(self, prediction, targets):
        # BCE Loss
        loss_bce = self.BCELoss(prediction, targets)

        # Connect Loss
        loss_con = self.Connect_Loss(prediction, targets)

        # Combine the losses with weights
        total_loss = self.bce_weight * loss_bce + self.con_weight * loss_con

        return total_loss


def torch_norm(data):
    _range = torch.max(data) - torch.min(data)
    return (data - torch.min(data)) / (_range + 1e-10)


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Connect_Loss(nn.Module):
    def __init__(self):
        super(Connect_Loss, self).__init__()

    def smoothness_penalty(self, tensor):
        # Compute gradient differences for smooth connectivity
        grad_x = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1])
        grad_y = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :])
        smoothness_loss = torch.mean(grad_x) + torch.mean(grad_y)

        return smoothness_loss

    def forward(self, img, lab):
        kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=img.dtype).cuda()
        img = K.morphology.opening(img, kernel)
        lab = K.morphology.opening(lab, kernel)

        # Apply a sigmoid to approximate binarization in a differentiable way
        img = torch.sigmoid(10 * (img - 0.5))  # Sharpens values close to 0 or 1
        lab = torch.sigmoid(10 * (lab - 0.5))  # Sharpens values close to 0 or 1

        smoothness_img = self.smoothness_penalty(img)
        smoothness_lab = self.smoothness_penalty(lab)

        C = torch.abs(smoothness_img - smoothness_lab) / torch.count_nonzero(lab)

        return C

# 结合使用BCEWithLogitsLoss和Dice Loss损失函数，alpha可以调整两种函数权重
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1., alpha=0.5):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # 计算BCEWithLogitsLoss
        bce_loss = self.bce_loss(inputs, targets)

        # 计算Dice Loss
        inputs = torch.sigmoid(inputs)  # 确保inputs在0到1之间
        intersection = (inputs * targets).sum(dim=[1, 2, 3])
        dice = (2. * intersection + self.smooth) / (
                    inputs.sum(dim=[1, 2, 3]) + targets.sum(dim=[1, 2, 3]) + self.smooth)
        dice_loss = 1 - dice.mean()

        # 组合两种损失
        loss = self.alpha * bce_loss + (1 - self.alpha) * dice_loss
        return loss