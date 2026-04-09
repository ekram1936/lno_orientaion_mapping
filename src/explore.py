"""
Data exploration and visualization for the LiNiO₂ orientation dataset. 
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

EXPLORE_DIR = "outputs/exploration"


def _save(fig, name):
    os.makedirs(EXPLORE_DIR, exist_ok=True)
    path = os.path.join(EXPLORE_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# Euler Angle Distributions
def plot_angle_distributions(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Orientation Distribution Analysis",
                 fontweight="bold", fontsize=13)
    specs = [
        ("phi1", "φ₁ (degrees)", "steelblue"),
        ("Phi",  "Φ (degrees)",  "darkorange"),
        ("phi2", "φ₂ (degrees)", "seagreen"),
    ]
    for ax, (col, xlabel, color) in zip(axes, specs):
        ax.hist(df[col], bins=60, color=color,
                edgecolor="white", linewidth=0.3)
        mean_val = df[col].mean()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5,
                   label=f"mean={mean_val:.1f}°")
        ax.legend(fontsize=9, loc="upper right")
        ax.set_title(f"Distribution of {col}  (degrees)", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, "angle_distributions.png")


# Joint angle distributions
def plot_angle_correlations(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [
        ("phi1", "Phi"),
        ("phi1", "phi2"),
        ("Phi",  "phi2"),
    ]
    for ax, (a, b) in zip(axes, pairs):
        hb = ax.hexbin(df[a], df[b], gridsize=40, cmap="YlOrRd", mincnt=1)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Count", fontsize=9)
        ax.set_xlabel(f"{a} (°)", fontsize=10)
        ax.set_ylabel(f"{b} (°)", fontsize=10)
        ax.set_title(f"Joint: {a} (°) vs {b} (°)", fontsize=10)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    _save(fig, "angle_correlations.png")


# Dataset overview
def plot_dataset_overview(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Unique Angle Values", fontweight="bold", fontsize=13)
    specs = [
        ("phi1", "φ₁ (degrees)", "steelblue"),
        ("Phi",  "Φ (degrees)",  "darkorange"),
        ("phi2", "φ₂ (degrees)", "seagreen"),
    ]
    for ax, (col, xlabel, color) in zip(axes, specs):
        uv = sorted(df[col].unique())
        counts = df.groupby(col).size().reindex(uv).values
        mean_val = df[col].mean()
        ax.bar(uv, counts, color=color, alpha=0.85,
               width=max(np.diff(uv).min() * 0.9, 1.0))
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5,
                   label=f"mean={mean_val:.1f}°")
        ax.legend(fontsize=9, loc="upper right")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {col}  (degrees)", fontsize=10)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, "dataset_overview.png")


# Sample diffraction images
def plot_sample_images(df, image_dir, n=9):
    sample = df.sample(n=n, random_state=42)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
    for ax, (_, row) in zip(axes.flat, sample.iterrows()):
        img = np.array(
            Image.open(os.path.join(image_dir, row["filename"])).convert("L"),
            dtype=np.float32,
        )
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"φ₁={row['phi1']:.1f}°Φ={row['Phi']:.2f}°  φ₂={row['phi2']:.1f}°",
            fontsize=7, color="white",
        )
        ax.set_facecolor("black")
        ax.axis("off")
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.patch.set_facecolor("white")
    plt.tight_layout(pad=0.5)
    _save(fig, "sample_images.png")


# Pixel statistics across N random images────
def plot_pixel_statistics(df, image_dir, n=200):
    sample = df.sample(n=n, random_state=0)
    means, stds, mins, maxs = [], [], [], []
    for _, row in sample.iterrows():
        img = np.array(
            Image.open(os.path.join(image_dir, row["filename"])).convert("L"),
            dtype=np.float32,
        )
        means.append(img.mean())
        stds.append(img.std())
        mins.append(img.min())
        maxs.append(img.max())

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(
        f"Pixel Statistics Across {n} Random Images", fontsize=13, fontweight="bold")
    for ax, data, label, color in zip(
        axes,
        [means, stds, mins, maxs],
        ["Pixel Mean", "Pixel Std Dev", "Pixel Min", "Pixel Max"],
        ["steelblue", "darkorange", "seagreen", "tomato"],
    ):
        ax.hist(data, bins=30, color=color, edgecolor="white", linewidth=0.4)
        ax.axvline(np.mean(data), color="black", linestyle="--", linewidth=1.5,
                   label=f"mean={np.mean(data):.1f}")
        ax.legend(fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Value", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, "pixel_statistics.png")

    # print summary
    print(f"Pixel Statistics Summary ({n} images)")
    print(f"  Mean of means : {np.mean(means):.2f}")
    print(f"  Mean of stds  : {np.mean(stds):.2f}")
    print(f"  Typical min   : {np.mean(mins):.2f}")
    print(f"  Typical max   : {np.mean(maxs):.2f}")


# Single image: raw | log-scaled | pixel histogram
def plot_log_normalisation(df, image_dir, idx=0):
    row = df.iloc[idx]
    img = np.array(
        Image.open(os.path.join(image_dir, row["filename"])).convert("L"),
        dtype=np.float32,
    )
    log_img = np.log1p(img)
    log_img_vis = (log_img / log_img.max() *
                   255).astype(np.uint8)  # stretch for colormap

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    title_str = f"φ₁={row['phi1']:.1f}°  Φ={row['Phi']:.2f}°  φ₂={row['phi2']:.1f}°"

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Raw Image{title_str}", fontsize=9)
    axes[0].axis("off")

    from matplotlib.colors import LinearSegmentedColormap
    colors_cmap = ["black", "purple", "red", "orange", "yellow", "white"]
    cmap_fire = LinearSegmentedColormap.from_list("fire", colors_cmap)
    axes[1].imshow(log_img_vis, cmap=cmap_fire)
    axes[1].set_title("Log-Scaled(faint spots boosted)", fontsize=9)
    axes[1].axis("off")

    axes[2].hist(img.flatten(), bins=80,
                 color="steelblue", alpha=0.85, log=True)
    axes[2].set_xlabel("Pixel value (0-255)", fontsize=10)
    axes[2].set_ylabel("Count (log scale)", fontsize=10)
    axes[2].set_title("Pixel Intensity Histogram(log y-scale)", fontsize=9)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, "log_normalisation.png")


# Similar vs dissimilar orientations
def plot_similar_orientations(df, image_dir, n_close=3, n_far=3, random_state=7):
    rng = np.random.default_rng(random_state)
    ref_idx = int(rng.integers(len(df)))
    ref = df.iloc[ref_idx]

    euler_cols = ["phi1", "Phi", "phi2"]
    all_euler = df[euler_cols].values.astype(float)
    ref_euler = all_euler[[ref_idx]]

    dists = np.sqrt(((all_euler - ref_euler) ** 2).sum(axis=1))
    dists[ref_idx] = np.inf  # exclude self

    close_idx = np.argsort(dists)[:n_close]
    far_idx = np.argsort(dists)[-n_far:][::-1]

    def load(row):
        return np.array(
            Image.open(os.path.join(image_dir, row["filename"])).convert("L"),
            dtype=np.float32,
        )

    ncols = 1 + n_close
    fig, axes = plt.subplots(2, ncols, figsize=(ncols * 3.2, 7))
    fig.suptitle("Do Similar Orientations Produce Similar Patterns?",
                 fontsize=12, fontweight="bold")

    ref_row = df.iloc[ref_idx]
    axes[0, 0].imshow(load(ref_row), cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(
        f"REFERENCE φ₁={ref_row.phi1:.1f}° Φ={ref_row.Phi:.2f}°  φ₂={ref_row.phi2:.1f}°",
        fontsize=7, color="blue", fontweight="bold",
    )
    axes[0, 0].axis("off")
    axes[0, 0].set_facecolor("black")

    for k, cidx in enumerate(close_idx):
        crow = df.iloc[cidx]
        d = dists[cidx]
        axes[0, k + 1].imshow(load(crow), cmap="gray", vmin=0, vmax=255)
        Δ1 = abs(crow.phi1 - ref_row.phi1)
        ΔΦ = abs(crow.Phi - ref_row.Phi)
        Δ2 = abs(crow.phi2 - ref_row.phi2)
        axes[0, k + 1].set_title(
            f"CLOSE #{k+1} φ₁={crow.phi1:.1f}° (Δ{Δ1:.1f}°) Φ={crow.Phi:.2f}°  φ₂={crow.phi2:.1f}° (Δ{Δ2:.1f}°)",
            fontsize=7, color="green",
        )
        axes[0, k + 1].axis("off")
        axes[0, k + 1].set_facecolor("black")

    axes[1, 0].axis("off")

    for k, fidx in enumerate(far_idx):
        frow = df.iloc[fidx]
        Δ1 = abs(frow.phi1 - ref_row.phi1)
        ΔΦ = abs(frow.Phi - ref_row.Phi)
        Δ2 = abs(frow.phi2 - ref_row.phi2)
        axes[1, k + 1].imshow(load(frow), cmap="gray", vmin=0, vmax=255)
        axes[1, k + 1].set_title(
            f"FAR  # {k+1} φ₁={frow.phi1: .1f}° (Δ{Δ1: .1f}°) Φ={frow.Phi: .2f}°  φ₂={frow.phi2: .1f}°",
            fontsize=7, color="red",
        )
        axes[1, k + 1].axis("off")
        axes[1, k + 1].set_facecolor("black")

    plt.tight_layout(pad=0.8)
    _save(fig, "similar_orientations.png")


def print_stats(df):
    sep = "=" * 60
    print(sep)
    print("DATA EXPLORATION")
    print(sep)
    print()
    print("DATASET")
    print(f"  Total images    : {len(df):,}")
    print(f"  Columns         : {list(df.columns)}")
    print()
    print("ANGLE RANGES")
    for col in ["phi1", "Phi", "phi2"]:
        print(f"  {col:5s}  min={df[col].min():.1f}°  max={df[col].max():.1f}°  "
              f"mean={df[col].mean():.1f}°  std={df[col].std():.1f}°  "
              f"nunique={df[col].nunique()}")
    print()
    print(sep)


def run_all_exploration(df, image_dir):
    print_stats(df)
    print("\n-- Data Exploration ----------------------")
    plot_angle_distributions(df)
    plot_angle_correlations(df)
    plot_dataset_overview(df)
    plot_sample_images(df, image_dir)
    plot_pixel_statistics(df, image_dir)
    plot_log_normalisation(df, image_dir)
    plot_similar_orientations(df, image_dir)
    print("-- Exploration complete ------------------\n")
