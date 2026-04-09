"""
Loss functions for training the orientation model."""
import torch
import torch.nn as nn


class SymmetryAwareLoss(nn.Module):
    """
    Symmetry-aware loss for φ₁ with 6-fold symmetry.  For each sample, computes MSE
    """

    def __init__(self, symmetry_step_deg=60.0):
        super().__init__()
        steps = torch.arange(0, 360, symmetry_step_deg)
        self.register_buffer("steps_rad", torch.deg2rad(steps))

    def forward(self, pred, target):
        loss_others = ((pred[:, 2:] - target[:, 2:]) ** 2).mean()
        true_phi1 = torch.atan2(target[:, 0], target[:, 1])
        equiv = true_phi1.unsqueeze(1) + self.steps_rad
        loss_per = (pred[:, 0:1] - torch.sin(equiv))**2 + \
            (pred[:, 1:2] - torch.cos(equiv))**2
        return loss_per.min(dim=1).values.mean() + loss_others * 2.0


def build_loss(cfg: dict) -> nn.Module:
    t = cfg["loss"]["type"]
    if t == "SymmetryAwareLoss":
        fold = cfg["loss"].get("loss_fold_deg", 60.0)
        print(f"  Loss: SymmetryAwareLoss (fold={fold} deg, "
              f"generates {int(360/fold)} equivalents)")
        return SymmetryAwareLoss(fold)
    print("  Loss: MSELoss")
    return nn.MSELoss()
