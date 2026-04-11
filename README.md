# LNO Orientation Mapping

<img src="asset/arc.png" alt="Pipeline Architecture" width="500"/>

> **Deep Learning for Orientation Mapping in 4D-STEM**  
> Md Ekram Hossain | PhD Candidate | Erlangen, Germany | ekramml0013@gmail.com

---

## About

A 4D-STEM experiment generates hundreds of thousands of diffraction images, each encoding
the crystal orientation of a tiny grain. This pipeline trains a **4-block Convolutional
Neural Network (OrientationCNN, ~1.2M params)** to directly predict three Bunge–Euler
angles (φ₁, Φ, φ₂) from 256×256 grayscale LiNiO₂ (LNO) diffraction patterns.

Euler angles are encoded as **sin/cos pairs** (6 outputs) to handle angle periodicity,
and decoded via `atan2` after prediction. A fourth experiment explores **quaternion
representations** (4 outputs) as an alternative encoding. Four experiments progressively
address the challenges of symmetry, data scale, and output space design.

See [PhD_Coding_Task_Report.pdf](report/PhD_Coding_Task_Report.pdf) for full methodology,
analysis and results.

---

## Key Results

| Experiment | Samples | Loss                     | Epochs | φ₁ MAE (°) | Φ MAE (°) | φ₂ MAE (°) | Overall MAE (°) |
| ---------- | ------- | ------------------------ | ------ | ---------- | --------- | ---------- | --------------- |
| Exp 1      | 10k     | MSELoss                  | 20     | 88.60      | 1.75      | 4.82       | 31.72           |
| Exp 2      | 30k     | SymmetryAwareLoss        | 50     | 88.52      | 1.28      | 2.58       | 30.79           |
| **Exp 3**  | **10k** | **MSELoss + φ₁ folding** | **30** | **14.83**  | **0.77**  | **2.14**   | **5.91**        |
| Exp 4      | 10k     | QuaternionLoss           | 30     | 90.59      | 4.14      | 15.34      | 36.69           |

> **φ₁ cannot be learned** due to the 6-fold hexagonal symmetry of LiNiO₂ — six
> different φ₁ values produce identical diffraction patterns. This is a physical
> limitation of the data, not the model. **Φ and φ₂ are reliably learnable**,
> with best MAE of 0.77° and 2.14° respectively (Experiment 3).Experiment 4
> confirms this finding holds even with quaternion representations.

---

## Dataset

| Property     | Value                                                             |
| ------------ | ----------------------------------------------------------------- |
| Source       | Zenodo record [17360572](https://doi.org/10.5281/zenodo.17360572) |
| Total images | 581,328 simulated LiNiO₂ patterns                                 |
| Image size   | 256×256 px, grayscale PNG                                         |
| Labels       | Bunge–Euler angles parsed from filenames                          |
| φ₁ range     | [0°, 360°], step 2.5°                                             |
| Φ range      | [0°, 90°]                                                         |
| φ₂ range     | [60°, 120°]                                                       |

> **Preprocessing**: Log-scaling `x → ln(1+x)/ln(256)` is applied — 94% of pixels
> have intensity below 5/255 so raw values are very sparse.

---

## Quick Start

```bash
# 1. Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt

# 2. Download and extract dataset (~4.7 GB from Zenodo)
python main.py --download-only

# 3. Explore data — all plots saved to outputs/exploration/
python explore.py

# 4. Run one experiment
python main.py --config config/config_exp1.yaml

# 5. Run all 4 experiments + comparison
python main.py --all
```

> After running, `outputs/` will be populated with checkpoints, training curves,
> scatter plots, and results per experiment (see structure below).

---

## Project Structure

```bash
lno_orientation/
├── config/
│ ├── config_exp1.yaml ← 10k, MSELoss, 20 epochs
│ ├── config_exp2.yaml ← 30k, SymmetryAwareLoss, 50 epochs
│ └── config_exp3.yaml ← 10k, MSELoss, 50 epochs
  └── config_exp4.yaml  ← 10k, QuaternionLoss, 30 epochs
├── data/
│ ├── raw/ ← downloaded zip files
│ ├── images/ ← extracted PNG images
│ └── labels.csv ← parsed from filenames
├── src/
│ ├── utils.py ← encode/decode/angular_error/seeds/device
│ ├── dataset.py ← download / extract / parse labels
│ ├── dataloader.py ← subsample + split + DiffractionDataset
│ ├── model.py ← OrientationCNN (4-block CNN, ~1.2M params)
│ ├── loss.py ← MSELoss + SymmetryAwareLoss
│ ├── train.py ← training loop + early stopping
│ ├── evaluate.py ← test MAE + results.json + accuracy_table
│ ├── visualise.py ← curves / scatter / best-worst / comparison
│ └── explore.py ← exploration plots
├── outputs/
│ ├── exploration/ ← Euler angle distributions, joint plots, log-scaling effect
│ ├── experiment_1/ ← checkpoints / splits / plots / visuals / results.json
│ ├── experiment_2/
│ ├── experiment_3/
  ├── experiment_4/
│ └── comparison/ ← comparison_bar.png + comparison_table.csv
├── main.py ← full pipeline entry point
├── explore.py ← standalone exploration script
└── requirements.txt
```

---

## Outputs per Experiment

```bash
outputs/experiment_N/
├── checkpoints/best_model.pth
├── splits/train.csv, val.csv, test.csv ← 80/10/10 split
├── plots/training_curves.png
├── plots/scatter_pred_vs_true.png ← φ₁, Φ, φ₂ predicted vs true
├── plots/accuracy_table.png
├── plots/training_history.csv
├── visuals/best_worst_predictions.png
└── results.json ← MAE per angle + overall
```

---

## Model Architecture

| Layer       | Output Shape      | Description                                   |
| ----------- | ----------------- | --------------------------------------------- |
| Input       | (B, 1, 256, 256)  | Log-scaled grayscale image                    |
| ConvBlock 1 | (B, 32, 128, 128) | Conv-BN-ReLU, MaxPool                         |
| ConvBlock 2 | (B, 64, 64, 64)   | Conv-BN-ReLU, MaxPool                         |
| ConvBlock 3 | (B, 128, 32, 32)  | Conv-BN-ReLU, MaxPool                         |
| ConvBlock 4 | (B, 256, 16, 16)  | Conv-BN-ReLU, MaxPool                         |
| GAP         | (B, 256)          | Global Average Pooling                        |
| FC1         | (B, 512)          | Linear-ReLU-Dropout(0.3)                      |
| FC2         | (B, 256)          | Linear-ReLU-Dropout(0.3)                      |
| Output      | (B, 6) or (B, 4)  | sin/cos pairs (Exp 1–3) or quaternion (Exp 4) |

---

## Changing Experiment Parameters

Edit any `config/config_expN.yaml`. All parameters live there:

```yaml
sample_size: 10000
epochs: 20
lr: 0.001
loss: mse # or symmetry_aware
scheduler: step # or cosine
batch_size: 32
```

---

## References

1. J. Scheunert, _CNNs for Orientation Mapping of LNO Electron Diffraction Patterns Together with Test-data_, Zenodo, Oct. 2025. DOI: [10.5281/zenodo.17360572](https://doi.org/10.5281/zenodo.17360572)
2. J. Scheunert, S. Ahmed, T. Demuth, A. Beyer, S. Wissel, B.-X. Xu, and K. Volz, _Determining the grain orientations of battery materials from electron diffraction patterns using convolutional neural networks_, npj Computational Materials, 2026.
3. E. F. Rauch and M. Véron, _Automated crystal orientation and phase mapping in TEM_, Materials Characterization, vol. 98, pp. 1–9, 2014. DOI: [10.1016/j.matchar.2014.08.010](https://doi.org/10.1016/j.matchar.2014.08.010)
4. A. Rougier, P. Gravereau, and C. Delmas, _Optimization of the Composition of the Li₁₋ᵤNi₁₊ᵤO₂ Electrode Materials: Structural, Magnetic, and Electrochemical Studies_, Journal of The Electrochemical Society, vol. 143, no. 4, pp. 1168–1175, 1996. DOI: [10.1149/1.1836614](https://doi.org/10.1149/1.1836614)
5. Y. Zhou, C. Barnes, J. Lu, J. Yang, and L. Li, _On the Continuity of Rotation Representations in Neural Networks_, in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5745–5753, 2019. DOI: [10.1109/CVPR.2019.00589](https://doi.org/10.1109/CVPR.2019.00589)
