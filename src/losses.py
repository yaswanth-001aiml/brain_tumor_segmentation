import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Dice Loss
# ============================================================

class DiceLoss(nn.Module):
    """
    Multiclass Soft Dice Loss.
    Ignores background (class 0) by default to focus on tumor subregions.
    """

    def __init__(self, smooth=1e-5, include_background=False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits, target):
        probabilities = torch.softmax(logits, dim=1)
        num_classes = logits.shape[1]

        target_one_hot = F.one_hot(target, num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()

        if not self.include_background and num_classes > 1:
            probabilities = probabilities[:, 1:]
            target_one_hot = target_one_hot[:, 1:]

        intersection = (probabilities * target_one_hot).sum(dim=(2, 3, 4))
        denominator = (
            probabilities.sum(dim=(2, 3, 4)) +
            target_one_hot.sum(dim=(2, 3, 4))
        )

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


# ============================================================
# Generalized Dice Loss
# ============================================================

class GeneralizedDiceLoss(nn.Module):
    """
    Generalized Dice Loss (GDL) to tackle extreme class imbalance.
    Weights each class inversely proportional to label frequency.
    """

    def __init__(self, smooth=1e-5, include_background=False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits, target):
        probabilities = torch.softmax(logits, dim=1)
        num_classes = logits.shape[1]

        target_one_hot = F.one_hot(target, num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()

        if not self.include_background and num_classes > 1:
            probabilities = probabilities[:, 1:]
            target_one_hot = target_one_hot[:, 1:]

        # Weighting: 1 / (volume^2)
        volumes = target_one_hot.sum(dim=(2, 3, 4))
        weights = 1.0 / (volumes ** 2 + self.smooth)

        # Normalize weights across classes
        weights = weights / (weights.sum(dim=1, keepdim=True) + self.smooth)

        intersection = (probabilities * target_one_hot).sum(dim=(2, 3, 4))
        denominator = probabilities.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))

        numerator = 2.0 * (weights * intersection).sum(dim=1) + self.smooth
        denom = (weights * denominator).sum(dim=1) + self.smooth

        gdl = numerator / denom
        return 1.0 - gdl.mean()


# ============================================================
# Focal Loss
# ============================================================

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss to down-weight easy background examples.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, target):
        ce_loss = F.cross_entropy(
            logits, target, weight=self.weight, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ============================================================
# Combined Dice + Cross-Entropy Loss
# ============================================================

class CombinedLoss(nn.Module):
    """
    Standard combination of Soft Dice Loss and Cross-Entropy Loss.
    loss = dice_weight * DiceLoss + ce_weight * CrossEntropy
    """

    def __init__(self, dice_weight=1.0, ce_weight=1.0, class_weights=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, target):
        dice = self.dice_loss(logits, target)
        ce = self.cross_entropy(logits, target)
        return self.dice_weight * dice + self.ce_weight * ce


# ============================================================
# Focal + Dice Loss
# ============================================================

class FocalDiceLoss(nn.Module):
    """
    Combination of Soft Dice Loss and Focal Loss.
    """

    def __init__(self, dice_weight=1.0, focal_weight=1.0, gamma=2.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma=gamma)

    def forward(self, logits, target):
        dice = self.dice_loss(logits, target)
        focal = self.focal_loss(logits, target)
        return self.dice_weight * dice + self.focal_weight * focal


# ============================================================
# Loss Factory
# ============================================================

LOSS_REGISTRY = {
    "combined": CombinedLoss,
    "dice": DiceLoss,
    "focal": FocalLoss,
    "focal_dice": FocalDiceLoss,
    "generalized_dice": GeneralizedDiceLoss,
    "ce": nn.CrossEntropyLoss,
}


def get_loss(loss_name="combined", **kwargs):
    """
    Loss factory helper.
    Supported: 'combined', 'dice', 'focal', 'focal_dice', 'generalized_dice', 'ce'
    """
    key = loss_name.lower().strip()
    if key not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss type: '{loss_name}'. Available: {list(LOSS_REGISTRY.keys())}"
        )
    return LOSS_REGISTRY[key](**kwargs)