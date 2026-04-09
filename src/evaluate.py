"""evaluate.py – test-set inference, metrics, save results.json + accuracy_table.csv."""
import os
import json
import numpy as np
import pandas as pd
import torch
from src.utils import decode_sincos, angular_error


def evaluate(model, test_loader, df_test, cfg, device) -> dict:
    model.eval()
    preds_l, labels_l = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            preds_l.append(model(imgs.to(device)).cpu().numpy())
            labels_l.append(labels.numpy())
    step = cfg["loss"].get("symmetry_step_deg", 360.0)
    pred_a = decode_sincos(np.vstack(preds_l), step=step)
    true_a = decode_sincos(np.vstack(labels_l), step=step)
    per_mae, overall = angular_error(pred_a, true_a, step=step)

    results = {
        "experiment":        cfg["experiment"]["name"],
        "sample_size":       cfg["data"]["sample_size"],
        "test_mae_phi1_deg": round(float(per_mae[0]),  4),
        "test_mae_Phi_deg":  round(float(per_mae[1]),  4),
        "test_mae_phi2_deg": round(float(per_mae[2]),  4),
        "test_mae_overall":  round(float(overall), 4),
    }
    out = cfg["experiment"]["output_dir"]
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    rows = [
        {"Angle": "phi1",    "Test MAE (deg)": results["test_mae_phi1_deg"]},
        {"Angle": "Phi",     "Test MAE (deg)": results["test_mae_Phi_deg"]},
        {"Angle": "phi2",    "Test MAE (deg)": results["test_mae_phi2_deg"]},
        {"Angle": "Overall", "Test MAE (deg)": results["test_mae_overall"]},
    ]
    pd.DataFrame(rows).to_csv(os.path.join(
        out, "accuracy_table.csv"), index=False)

    print(f"\n  Test MAE  phi1={per_mae[0]:.2f}  Phi={per_mae[1]:.2f}  "
          f"phi2={per_mae[2]:.2f}  Overall={overall:.2f} deg")

    results["_pred_angles"] = pred_a
    results["_true_angles"] = true_a
    results["_df_test"] = df_test
    return results
