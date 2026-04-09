"""
Run data exploration independently (no training needed).
"""
import yaml
from src.dataset import parse_labels
from src.explore import run_all_exploration

if __name__ == "__main__":
    with open("config/config_exp1.yaml") as f:
        cfg = yaml.safe_load(f)
    df = parse_labels(cfg)
    run_all_exploration(df, cfg["data"]["image_dir"])
