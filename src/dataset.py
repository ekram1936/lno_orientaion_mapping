"""
Dataset handling: download, extract, parse labels.
"""
import os
import re
import zipfile
import requests
import pandas as pd
from tqdm import tqdm

DATA_CONFIG = {
    "zenodo_base_url": "https://zenodo.org/records/17360572/files",
    "dataset_zip":     "LNO_simulated_test_dataset.zip",
    "image_subfolder": "LNO_phi102_sample_res_1",
    "image_size":      256,
}


def _download_file(url: str, dest: str):
    if os.path.exists(dest):
        print(f"  [skip] {os.path.basename(dest)} already downloaded.")
        return
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"  Downloading {os.path.basename(dest)} ...")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"  Saved -> {dest}")


def download_dataset(cfg: dict):
    """Download only the image zip from Zenodo."""
    raw_dir = cfg["data"]["raw_dir"]
    base_url = DATA_CONFIG["zenodo_base_url"]
    os.makedirs(raw_dir, exist_ok=True)

    fname = DATA_CONFIG["dataset_zip"]
    url = f"{base_url}/{fname}?download=1"
    _download_file(url, os.path.join(raw_dir, fname))


def extract_dataset(cfg: dict):
    """
    Unzip all .zip files in raw_dir into image_dir.
    """
    raw_dir = cfg["data"]["raw_dir"]
    image_dir = cfg["data"]["image_dir"]
    os.makedirs(image_dir, exist_ok=True)

    zips = [f for f in os.listdir(raw_dir) if f.endswith(".zip")]
    if not zips:
        print(f"  No zip files found in {raw_dir}")
        return

    for zf in zips:
        zip_path = os.path.join(raw_dir, zf)

        # Skip if already extracted
        existing_pngs = [
            f for _, _, files in os.walk(image_dir)
            for f in files if f.endswith(".png")
        ]
        if existing_pngs:
            print(
                f"  [skip] Already extracted — found {len(existing_pngs):,} PNG(s) in {image_dir}")
            break

        print(f"  Extracting {zf} ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(image_dir)
        print(f"  Extraction complete -> {image_dir}")

    print(f"\n  Folder structure under {image_dir}:")
    for root, dirs, files in os.walk(image_dir):
        depth = root.replace(image_dir, "").count(os.sep)
        if depth > 1:
            continue
        indent = "    " + "  " * depth
        print(f"{indent}{os.path.basename(root)}/  ({len(files)} files)")
        if depth == 1:
            dirs.clear()

    print(f"\n  Sample filenames (for pattern check):")
    count = 0
    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.endswith(".png"):
                print(
                    f"    {os.path.relpath(os.path.join(root, f), image_dir)}")
                count += 1
            if count >= 3:
                break
        if count >= 3:
            break
    if count == 0:
        print("    WARNING: No PNG files found after extraction!")
        print("    Check the zip contents — it may contain a nested subfolder.")


def parse_labels(cfg: dict) -> pd.DataFrame:
    """
    Parse φ₁, Φ, φ₂ from filenames and save to labels_csv.
    """
    image_dir = cfg["data"]["image_dir"]
    labels_csv = cfg["data"]["labels_csv"]

    if os.path.exists(labels_csv):
        df = pd.read_csv(labels_csv)
        print(f"  [skip] Loaded {len(df):,} labels from {labels_csv}")
        return df

    pattern = re.compile(
        r"image_phi1_([\d.]+)_phi_([\d.]+)_phi2_([\d.]+)_thickness_\d+\.png", re.IGNORECASE)
    records = []
    for root, _, fnames in os.walk(image_dir):
        for fname in fnames:
            if not fname.lower().endswith(".png"):
                continue
            m = pattern.search(fname)
            if m:
                records.append({
                    "filename": os.path.relpath(os.path.join(root, fname), image_dir),
                    "phi1": float(m.group(1)),
                    "Phi":  float(m.group(2)),
                    "phi2": float(m.group(3)),
                })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(labels_csv) or ".", exist_ok=True)
    df.to_csv(labels_csv, index=False)
    print(f"  Parsed {len(df):,} images -> {labels_csv}")
    return df
