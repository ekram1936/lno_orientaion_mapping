"""
train.py – training loop, validation, early stopping, checkpointing, history.
"""
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import decode_sincos, angular_error, quat_to_euler


def build_optimizer_and_scheduler(model, cfg):
    opt = optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    sched = cfg["training"]["scheduler"]
    if sched == "CosineAnnealingLR":
        s = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg["training"]["epochs"], eta_min=1e-6
        )
    else:
        s = optim.lr_scheduler.StepLR(
            opt,
            step_size=cfg["training"]["step_size"],
            gamma=cfg["training"]["gamma"],
        )
    print(f"  Optimizer : Adam  lr={cfg['training']['lr']}  "
          f"weight_decay={cfg['training']['weight_decay']}")
    print(f"  Scheduler : {sched}")
    return opt, s


def _train_epoch(model, loader, criterion, opt, device):
    model.train()
    total = 0.0
    for imgs, labels in loader:
        imgs = imgs.float().to(device)
        labels = labels.float().to(device)
        opt.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * imgs.size(0)
    return total / len(loader.dataset)


def _validate(model, loader, criterion, device, step: float = 360.0, encoding: str = 'sincos'):
    """
    Runs validation loop and computes both loss and angular MAE metrics.
    """
    model.eval()
    total, preds_l, labels_l = 0.0, [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.float().to(device)
            labels = labels.float().to(device)
            preds = model(imgs)
            total += criterion(preds, labels).item() * imgs.size(0)
            preds_l.append(preds.cpu().numpy())
            labels_l.append(labels.cpu().numpy())

    if encoding == 'quaternion':
        all_preds = quat_to_euler(np.vstack(preds_l))
        all_labels = quat_to_euler(np.vstack(labels_l))
        mae_step = 360.0
    else:
        all_preds = decode_sincos(np.vstack(preds_l),  step=step)
        all_labels = decode_sincos(np.vstack(labels_l), step=step)
        mae_step = step

    per_angle, overall_mae = angular_error(all_preds, all_labels, step=mae_step)
    return total / len(loader.dataset), overall_mae, per_angle


def run_training(model, train_loader, val_loader, criterion,
                 opt, scheduler, cfg, device):
    """
    Runs the full training loop with validation, early stopping, checkpointing, and history recording.
    """
    epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["early_stop_patience"]
    out_dir = cfg["experiment"]["output_dir"]
    step = cfg["loss"].get("symmetry_step_deg", 360.0)
    encoding = cfg.get('model', {}).get('encoding', 'sincos')

    best_weights_path = os.path.join(out_dir, "best_model.pth")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    history = {
        "train_loss": [], "val_loss": [],
        "val_mae":    [],                   # overall
        "phi1_mae":   [], "Phi_mae":  [], "phi2_mae": [],
        "lr":         [],
    }

    best_val, best_ep, no_imp = float("inf"), 0, 0
    t0 = time.time()

    W = 88
    print("=" * W)
    print(f"  {'Ep':>4}  {'Train':>10}  {'Val':>10}  "
          f"{'MAE':>8}  {'φ₁':>8}  {'Φ':>8}  {'φ₂':>8}  {'LR':>10}")
    print("  " + "-" * (W - 2))

    for ep in range(1, epochs + 1):

        tr = _train_epoch(model, train_loader, criterion, opt, device)
        vl, vm, per_angle_mae = _validate(
            model, val_loader, criterion, device, step=step, encoding=encoding)
        scheduler.step()
        lr = opt.param_groups[0]["lr"]

        history["train_loss"].append(tr)
        history["val_loss"].append(vl)
        history["val_mae"].append(vm)
        history["phi1_mae"].append(float(per_angle_mae[0]))
        history["Phi_mae"].append(float(per_angle_mae[1]))
        history["phi2_mae"].append(float(per_angle_mae[2]))
        history["lr"].append(lr)

        tag = ""
        if vl < best_val:
            best_val, best_ep, no_imp = vl, ep, 0
            torch.save(model.state_dict(), best_weights_path)
            tag = " <- best"
        else:
            no_imp += 1

        torch.save(
            {
                "epoch":             ep,
                "model_state":       model.state_dict(),
                "optimizer_state":   opt.state_dict(),
                "scheduler_state":   scheduler.state_dict(),
                "val_loss":          vl,
                "overall_mae":       vm,
                "per_angle_mae":     per_angle_mae.tolist(),
                "history":           history,
                "config":            cfg,
                "model_class":       model.__class__.__name__,
                "total_params":      sum(p.numel() for p in model.parameters()
                                         if p.requires_grad),
            },
            os.path.join(ckpt_dir, f"checkpoint_ep{ep:03d}.pth"),
        )

        print(f"  {ep:>4}  {tr:>10.6f}  {vl:>10.6f}  "
              f"{vm:>7.2f}°  {per_angle_mae[0]:>7.2f}°  "
              f"{per_angle_mae[1]:>7.2f}°  {per_angle_mae[2]:>7.2f}°  "
              f"{lr:>10.7f}{tag}")

        if patience and no_imp >= patience:
            print(f"\n  Early stopping (patience={patience}, "
                  f"no improvement since epoch {best_ep})")
            break

    elapsed = time.time() - t0
    print("=" * W)
    print(f"  Done in {elapsed / 60:.1f} min  |  "
          f"Best val loss {best_val:.6f} @ epoch {best_ep}")
    print(f"  Best weights : {best_weights_path}")
    print(f"  Checkpoints  : {ckpt_dir}/checkpoint_ep***.pth")

    summary = {
        "config": cfg,
        "model": {
            "class":        model.__class__.__name__,
            "total_params": sum(p.numel() for p in model.parameters()
                                if p.requires_grad),
        },
        "training": {
            "epochs_run":   len(history["train_loss"]),
            "best_epoch":   best_ep,
            "best_val_loss": best_val,
            "elapsed_min":  round(elapsed / 60, 2),
        },
        "best_val_metrics": {
            "overall_mae_deg": round(history["val_mae"][best_ep - 1], 4),
            "phi1_mae_deg":    round(history["phi1_mae"][best_ep - 1], 4),
            "Phi_mae_deg":     round(history["Phi_mae"][best_ep - 1], 4),
            "phi2_mae_deg":    round(history["phi2_mae"][best_ep - 1], 4),
        },
        "full_history": history,
        "outputs": {
            "best_weights":    best_weights_path,
            "checkpoints_dir": ckpt_dir,
            "resume_hint": (
                "ckpt = torch.load(path); "
                "model.load_state_dict(ckpt['model_state']); "
                "opt.load_state_dict(ckpt['optimizer_state']); "
                "scheduler.load_state_dict(ckpt['scheduler_state'])"
            ),
        },
    }
    summary_path = os.path.join(out_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary      : {summary_path}")
    print("=" * W)

    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    return history, best_ep
