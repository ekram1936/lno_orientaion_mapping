"""
main.py – Run the full pipeline from here.

Usage
-----
  python main.py --download-only                    # download + extract
  python main.py --explore-only                     # exploration plots only
  python main.py --config config/config_exp1.yaml   # one experiment
  python main.py --all                              # all 3 experiments + compare
"""

import argparse
import os
import yaml
import pandas as pd
import torch

from src.utils import set_seeds, get_device, make_dirs
from src.dataset import download_dataset, extract_dataset, parse_labels
from src.STEM_dataloader import subsample_and_split, build_dataloaders
from src.model import build_model
from src.loss import build_loss
from src.train import build_optimizer_and_scheduler, run_training
from src.evaluate import evaluate
from src.visualise import (plot_training_curves, plot_scatter,
                           plot_best_worst, plot_comparison)
from src.explore import run_all_exploration


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_history(history, cfg):
    plots_dir = os.path.join(cfg["experiment"]["output_dir"], "plots")
    os.makedirs(plots_dir, exist_ok=True)
    pd.DataFrame({
        "epoch":      range(1, len(history["train_loss"]) + 1),
        "train_loss": history["train_loss"],
        "val_loss":   history["val_loss"],
        "val_mae":    history["val_mae"],
        "lr":         history["lr"],
    }).to_csv(os.path.join(plots_dir, "training_history.csv"), index=False)


def step_download(cfg):
    print("\n[STEP 1] Download dataset from Zenodo")
    download_dataset(cfg)
    print("\n[STEP 2] Extract zip files")
    extract_dataset(cfg)


def step_parse_labels(cfg):
    print("\n[STEP 3] Parse labels from filenames")
    return parse_labels(cfg)


def step_explore(df, cfg):
    print("\n[STEP 4] Data exploration -> outputs/exploration/")
    run_all_exploration(df, cfg["data"]["image_dir"])


def step_run_experiment(cfg, device):
    name = cfg["experiment"]["name"]
    print(f"\n{'='*55}")
    print(f"  Experiment : {name}")
    print(f"  Sample     : {cfg['data']['sample_size']:,} images")
    print(f"  Epochs     : {cfg['training']['epochs']}")
    print(f"  Loss       : {cfg['loss']['type']}")
    print(f"{'='*55}")

    make_dirs(cfg)
    set_seeds(cfg["data"]["random_seed"])

    print("\n[A] Subsample + split")
    df_train, df_val, df_test = subsample_and_split(cfg)

    print("\n[B] Build DataLoaders")
    train_loader, val_loader, test_loader = build_dataloaders(
        df_train, df_val, df_test, cfg)

    print("\n[C] Build model")
    model = build_model(cfg, device)
    criterion = build_loss(cfg).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

    print("\n[D] Train")
    history, best_epoch = run_training(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler, cfg, device
    )
    save_history(history, cfg)

    print("\n[E] Evaluate on test set")
    results = evaluate(model, test_loader, df_test, cfg, device)

    print("\n[F] Generate plots")
    plot_training_curves(history, cfg)
    plot_scatter(results, cfg)
    plot_best_worst(results, cfg)

    print(f"\n  Done: {cfg['experiment']['output_dir']}/")
    return results, history, cfg


def main():
    parser = argparse.ArgumentParser(
        description="LNO Orientation Mapping Pipeline")
    parser.add_argument(
        "--config",        help="Path to a single experiment YAML")
    parser.add_argument("--all",           action="store_true",
                        help="Run all 3 experiments")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--explore-only",  action="store_true")
    args = parser.parse_args()

    device = get_device()
    base_cfg = load_config("config/config_exp1.yaml")
    print(f"\nDevice: {device}")

    step_download(base_cfg)
    df = step_parse_labels(base_cfg)
    print(f"  Total images available: {len(df):,}")

    if args.download_only:
        print("\nDownload complete. Exiting.")
        return

    step_explore(df, base_cfg)

    if args.explore_only:
        print("\nExploration complete. Exiting.")
        return

    configs_to_run = []
    if args.all:
        configs_to_run = [
            "config/config_exp1.yaml",
            "config/config_exp2.yaml",
            "config/config_exp3.yaml",
        ]
    elif args.config:
        configs_to_run = [args.config]
    else:
        print("\nNo experiment specified. Use --config or --all.")
        parser.print_help()
        return

    all_results, all_histories, all_cfgs = [], [], []
    for config_path in configs_to_run:
        cfg = load_config(config_path)
        results, history, cfg = step_run_experiment(cfg, device)
        all_results.append(results)
        all_histories.append(history)
        all_cfgs.append(cfg)

    if len(all_results) > 1:
        print("\n[G] Cross-experiment comparison -> outputs/comparison/")
        plot_comparison(all_results, all_histories, all_cfgs)

    print("\n All done!")


if __name__ == "__main__":
    main()
