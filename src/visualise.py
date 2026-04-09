"""Visualisation functions for training curves, scatter plots, and comparisons."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── training curves ───────────────────────────────────────────────────────────
def plot_training_curves(history, cfg):
    epochs = range(1, len(history["train_loss"]) + 1)
    d = os.path.join(cfg["experiment"]["output_dir"], "plots")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f'{cfg["experiment"]["name"]} – Training Curves', fontweight="bold")
    a1.plot(epochs, history["train_loss"], label="Train")
    a1.plot(epochs, history["val_loss"], label="Val")
    a1.set_xlabel("Epoch")
    a1.set_ylabel("Loss")
    a1.set_title("MSE Loss")
    a1.legend()
    a1.grid(alpha=0.3)
    a2.plot(epochs, history["val_mae"], color="darkorange")
    a2.set_xlabel("Epoch")
    a2.set_ylabel("MAE (deg)")
    a2.set_title("Val MAE (deg)")
    a2.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(d, "training_curves.png"))


# ── scatter predicted vs actual ───────────────────────────────────────────────
def plot_scatter(results, cfg):
    step = cfg["loss"].get("symmetry_step_deg", 360.0)
    angle_ranges = [step, 360.0, 360.0]

    pred, true = results["_pred_angles"], results["_true_angles"]
    d = os.path.join(cfg["experiment"]["output_dir"], "plots")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f'{cfg["experiment"]["name"]} – Predicted vs Actual', fontweight="bold")

    for i, (ax, name) in enumerate(zip(axes, ["phi1", "Phi", "phi2"])):
        rng = angle_ranges[i]
        half_range = rng / 2.0

        diff = np.abs(pred[:, i] - true[:, i])
        diff = np.where(diff > half_range, rng - diff,
                        diff)

        ax.scatter(true[:, i], pred[:, i], s=3, alpha=0.3, color="steelblue")
        ax.plot([0, rng], [0, rng], "r--", lw=1,
                label="Perfect")
        ax.set_xlim(0, rng)
        ax.set_ylim(0, rng)
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Pred {name}")
        ax.set_title(f"{name}  MAE={diff.mean():.2f} deg")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, os.path.join(d, "scatter_pred_vs_true.png"))

# ── best / worst predictions ─────────────────────────────────────────────────


def plot_best_worst(results, cfg, n=4):
    pred, true = results["_pred_angles"], results["_true_angles"]
    df_test = results["_df_test"]
    image_dir = cfg["data"]["image_dir"]
    vis_dir = os.path.join(cfg["experiment"]["output_dir"], "visuals")
    step = cfg["loss"].get("symmetry_step_deg", 360.0)
    angle_ranges = np.array([step, 360.0, 360.0])
    half_ranges = angle_ranges / 2.0
    diff = np.abs(pred - true)
    diff = np.where(diff > half_ranges, angle_ranges - diff, diff)
    err = diff.mean(axis=1)
    best_idx = np.argsort(err)[:n]
    worst_idx = np.argsort(err)[-n:][::-1]

    fig, axes = plt.subplots(n, 4, figsize=(18, n*4))
    fig.suptitle(f'{cfg["experiment"]["name"]} – Best (green) vs Worst (red)',
                 fontsize=13, fontweight="bold")

    for row, (b, w) in enumerate(zip(best_idx, worst_idx)):
        for col, (idx, label, color) in enumerate([(b, "BEST", "green"), (w, "WORST", "red")]):
            ax_img = axes[row, col*2]
            img = np.array(Image.open(os.path.join(
                image_dir, df_test.iloc[idx]["filename"])).convert("L"))
            ax_img.imshow(img, cmap="gray")
            t, p, e = true[idx], pred[idx], diff[idx]
            ax_img.set_title(f"{label} #{row+1}  err={err[idx]:.1f}\n"
                             f"True  phi1={t[0]:.1f} Phi={t[1]:.1f} phi2={t[2]:.1f}\n"
                             f"Pred  phi1={p[0]:.1f} Phi={p[1]:.1f} phi2={p[2]:.1f}",
                             fontsize=7, color=color, fontweight="bold")
            ax_img.axis("off")
            for sp in ax_img.spines.values():
                sp.set_edgecolor(color)
                sp.set_linewidth(2.5)
                sp.set_visible(True)

            ax_bar = axes[row, col*2+1]
            bc = ["green" if v < 5 else "orange" if v < 15 else "red" for v in e]
            ax_bar.barh(["phi1", "Phi", "phi2"], e,
                        color=bc, edgecolor="white")
            ax_bar.axvline(5,  color="green",  linestyle="--",
                           lw=1, alpha=0.7, label="5 deg")
            ax_bar.axvline(15, color="orange", linestyle="--",
                           lw=1, alpha=0.7, label="15 deg")
            ax_bar.set_xlabel("Error (deg)")
            ax_bar.set_title("Per-angle error", fontsize=8)
            ax_bar.legend(fontsize=7)
            ax_bar.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, os.path.join(vis_dir, "best_worst_predictions.png"))


# ── cross-experiment comparison ───────────────────────────────────────────────
def plot_comparison(all_results, all_histories, all_cfgs):
    comp = "outputs/comparison"
    os.makedirs(comp, exist_ok=True)
    colors = ["steelblue", "darkorange", "seagreen"]
    names = [r["experiment"] for r in all_results]

    fig, axes = plt.subplots(1, len(all_histories), figsize=(6 * len(all_histories), 4),
                             sharey=False)
    if len(all_histories) == 1:
        axes = [axes]
    fig.suptitle("Training vs Validation Loss — All Experiments",
                 fontweight="bold", fontsize=13)
    for ax, hist, name, color in zip(axes, all_histories, names, colors):
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax.plot(epochs, hist["train_loss"], label="Train", color=color,  lw=2)
        ax.plot(epochs, hist["val_loss"],   label="Val",   color=color,  lw=2,
                linestyle="--", alpha=0.7)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(comp, "all_train_val_loss.png"))

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle("Validation MAE per Epoch — All Experiments",
                 fontweight="bold", fontsize=13)
    for hist, name, color in zip(all_histories, names, colors):
        epochs = range(1, len(hist["val_mae"]) + 1)
        ax.plot(epochs, hist["val_mae"], label=name, color=color, lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val MAE (deg)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(comp, "all_val_mae_curves.png"))

    angle_keys = ["phi1_mae", "Phi_mae", "phi2_mae"]
    angle_names = ["φ₁", "Φ", "φ₂"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    fig.suptitle("Per-Angle Validation MAE per Epoch — All Experiments",
                 fontweight="bold", fontsize=13)
    for ax, key, aname in zip(axes, angle_keys, angle_names):
        for hist, name, color in zip(all_histories, names, colors):
            epochs = range(1, len(hist[key]) + 1)
            ax.plot(epochs, hist[key], label=name, color=color, lw=2)
        ax.set_title(f"{aname} MAE", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE (deg)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(comp, "all_per_angle_mae_curves.png"))

    keys = ["test_mae_phi1_deg", "test_mae_Phi_deg",
            "test_mae_phi2_deg", "test_mae_overall"]
    lnames = ["φ₁", "Φ", "φ₂", "Overall"]
    x = np.arange(len(lnames))
    w = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Test MAE Comparison — All Experiments",
                 fontweight="bold", fontsize=13)
    for i, (res, color, name) in enumerate(zip(all_results, colors, names)):
        vals = [res[k] for k in keys]
        bars = ax.bar(x + i * w, vals, w, label=name, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + w * (len(all_results) - 1) / 2)
    ax.set_xticklabels(lnames, fontsize=11)
    ax.set_ylabel("Test MAE (deg)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(comp, "comparison_bar.png"))

    # ── 5. Scatter: predicted vs true — all experiments side by side ──────────
    angle_labels = ["phi1", "Phi", "phi2"]
    fig, axes = plt.subplots(len(all_results), 3,
                             figsize=(15, 5 * len(all_results)))
    fig.suptitle("Predicted vs True Angles — All Experiments",
                 fontweight="bold", fontsize=13)
    if len(all_results) == 1:
        axes = [axes]
    for row, (res, cfg, color, name) in enumerate(
            zip(all_results, all_cfgs, colors, names)):
        pred = res["_pred_angles"]
        true = res["_true_angles"]
        step = cfg["loss"].get("symmetry_step_deg", 360.0)
        angle_ranges = [step, 360.0, 360.0]
        for col, (aname, rng) in enumerate(zip(angle_labels, angle_ranges)):
            ax = axes[row][col]
            half = rng / 2.0
            diff = np.abs(pred[:, col] - true[:, col])
            diff = np.where(diff > half, rng - diff, diff)
            ax.scatter(true[:, col], pred[:, col],
                       s=3, alpha=0.25, color=color)
            ax.plot([0, rng], [0, rng], "r--", lw=1, label="Perfect")
            ax.set_xlim(0, rng)
            ax.set_ylim(0, rng)
            ax.set_xlabel(f"True {aname} (deg)", fontsize=8)
            ax.set_ylabel(f"Pred {aname} (deg)", fontsize=8)
            ax.set_title(f"{name} | {aname}  MAE={diff.mean():.2f}°",
                         fontsize=9, fontweight="bold")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(comp, "all_scatter_pred_vs_true.png"))

    # ── 6. Summary table─────────────────────────────────────
    rows = []
    for res, hist in zip(all_results, all_histories):
        best_ep = int(np.argmin(hist["val_loss"])) + 1
        rows.append({
            "Experiment":        res["experiment"],
            "Sample Size":       res["sample_size"],
            "Best Epoch":        best_ep,
            "Best Val Loss":     round(min(hist["val_loss"]), 6),
            "Best Val MAE (°)":  round(hist["val_mae"][best_ep - 1], 4),
            "Test φ₁ MAE (°)":   res["test_mae_phi1_deg"],
            "Test Φ MAE (°)":    res["test_mae_Phi_deg"],
            "Test φ₂ MAE (°)":   res["test_mae_phi2_deg"],
            "Test Overall (°)":  res["test_mae_overall"],
        })
    df_summary = pd.DataFrame(rows)
    csv_path = os.path.join(comp, "comparison_table.csv")
    df_summary.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("  EXPERIMENT COMPARISON SUMMARY")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)
    print(f"  Saved {csv_path}")
