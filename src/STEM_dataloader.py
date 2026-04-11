"""
STEM Diffraction Dataset and Dataloader
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils import encode_sincos, euler_to_quat, set_seeds


def subsample_and_split(cfg: dict):
    """
    Subsample the dataset to n samples and split into train/val/test.
    """
    seed = cfg["data"]["random_seed"]
    n = cfg["data"]["sample_size"]
    splits_dir = os.path.join(cfg["experiment"]["output_dir"], "splits")
    os.makedirs(splits_dir, exist_ok=True)

    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "val.csv")
    test_csv = os.path.join(splits_dir, "test.csv")

    if os.path.exists(train_csv):
        print(f"  [skip] Splits already exist in {splits_dir}")
        return pd.read_csv(train_csv), pd.read_csv(val_csv), pd.read_csv(test_csv)

    set_seeds(seed)
    df = pd.read_csv(cfg["data"]["labels_csv"])
    df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    n_train = int(n * cfg["data"]["train_split"])
    n_val = int(n * cfg["data"]["val_split"])
    idx = np.random.permutation(n)

    df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
    df_val = df.iloc[idx[n_train:n_train + n_val]].reset_index(drop=True)
    df_test = df.iloc[idx[n_train + n_val:]].reset_index(drop=True)

    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv,     index=False)
    df_test.to_csv(test_csv,   index=False)
    print(
        f"  Train {len(df_train):,} | Val {len(df_val):,} | Test {len(df_test):,}")
    return df_train, df_val, df_test


class DiffractionDataset(Dataset):
    """
    Grayscale diffraction PNGs with log1p normalisation and optional flip augmentation.
    """
    LOG_MAX = np.log1p(255.0)

    def __init__(self, dataframe: pd.DataFrame, image_dir: str, augment: bool = False, symmetry_step: float = 360.0, encoding: str = 'sincos'):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.augment = augment
        self.symmetry_step = symmetry_step
        self.encoding = encoding
        angles = self.df[["phi1", "Phi", "phi2"]].values.astype(np.float32)
        if self.encoding == 'sincos':
            self.labels = encode_sincos(
                angles, step=symmetry_step).astype(np.float32)
        elif self.encoding == 'quaternion':
            self.labels = euler_to_quat(angles)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fname = self.df.loc[idx, "filename"]
        img = np.array(Image.open(os.path.join(self.image_dir, fname)).convert("L"),
                       dtype=np.float32)
        img = np.log1p(img) / self.LOG_MAX
        if self.augment:
            if np.random.rand() > 0.5:
                img = np.fliplr(img).copy()
            if np.random.rand() > 0.5:
                img = np.flipud(img).copy()
        return torch.tensor(img).unsqueeze(0), torch.tensor(self.labels[idx])


def build_dataloaders(df_train, df_val, df_test, cfg: dict):
    image_dir = cfg["data"]["image_dir"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    step = cfg["loss"].get("symmetry_step_deg", 360.0)
    encoding = cfg.get('model', {}).get('encoding', 'sincos')
    train_loader = DataLoader(DiffractionDataset(df_train, image_dir, augment=True, symmetry_step=step, encoding=encoding),
                              batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(DiffractionDataset(df_val,   image_dir, augment=False, symmetry_step=step, encoding=encoding),
                            batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(DiffractionDataset(df_test,  image_dir, augment=False, symmetry_step=step, encoding=encoding),
                             batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
