"""utils.py – encoding, angular error, seeds, device, directories."""
import os
import random
import numpy as np
import torch
import scipy


def reduce_phi1(phi1, step: float = 60.0) -> np.ndarray:
    """
    Reduce φ₁ by symmetry step.  E.g. for step=60°,  φ₁=75° → 15°.
    """
    return np.asarray(phi1, dtype=np.float32) % np.float32(step)


def encode_sincos(angles_deg: np.ndarray, step: float = 360.0) -> np.ndarray:
    """
    Encode Euler angles (N,3) in degrees → (N,6) sin-cos.  φ₁ is reduced by symmetry.
    """
    angles_deg = angles_deg.astype(np.float32)
    phi1_r = reduce_phi1(angles_deg[:, 0], step)
    rad1 = np.radians(phi1_r)
    radPhi = np.radians(angles_deg[:, 1])
    rad2 = np.radians(angles_deg[:, 2])
    return np.column_stack([
        np.sin(rad1),  np.cos(rad1),
        np.sin(radPhi), np.cos(radPhi),
        np.sin(rad2),  np.cos(rad2),
    ]).astype(np.float32)


def decode_sincos(sincos: np.ndarray, step: float = 360.0) -> np.ndarray:
    """(N,6) sin-cos → (N,3) degrees.
    """
    sincos = sincos.astype(np.float32)
    angles = np.zeros((sincos.shape[0], 3), dtype=np.float32)

    angles[:, 0] = (
        np.degrees(np.arctan2(sincos[:, 0], sincos[:, 1])).astype(np.float32)
        % np.float32(step)
    )
    for i, (s, c) in enumerate([(2, 3), (4, 5)], start=1):
        angles[:, i] = (
            np.degrees(np.arctan2(sincos[:, s], sincos[:, c])).astype(
                np.float32)
            % 360.0
        )
    return angles


def angular_error(pred: np.ndarray, true: np.ndarray,
                  step: float = 360.0):
    """
    Wrap-aware angular error.

    Args:
        pred, true: both shape (N,3) in degrees.
        step:       symmetry step for φ₁.
                    - 360.0 → Exp 1 & 2: wrap threshold = 180°
                    - 60.0  → Exp 3:     wrap threshold = 30°  (= step/2)

    Returns:
        per_angle_mae  shape (3,)  — MAE per [φ₁, Φ, φ₂]
        overall_mae    float       — mean over all angles & samples
    """
    pred = pred.astype(np.float32)
    true = true.astype(np.float32)

    diff_phi1 = np.abs(pred[:, 0] - true[:, 0])
    half_step = np.float32(step / 2.0)
    diff_phi1 = np.where(diff_phi1 > half_step,
                         np.float32(step) - diff_phi1,
                         diff_phi1).astype(np.float32)

    diff_others = np.abs(pred[:, 1:] - true[:, 1:])
    diff_others = np.where(diff_others > 180.0,
                           360.0 - diff_others,
                           diff_others).astype(np.float32)

    diff = np.column_stack([diff_phi1, diff_others]).astype(np.float32)
    return diff.mean(axis=0), float(diff.mean())


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_dirs(cfg: dict):
    """Create all output subdirectories for an experiment."""
    base = cfg["experiment"]["output_dir"]
    for sub in ["checkpoints", "splits", "plots", "visuals"]:
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    for d in [cfg["data"]["raw_dir"], cfg["data"]["image_dir"],
              "outputs/exploration", "outputs/comparison"]:
        os.makedirs(d, exist_ok=True)

# ── Experiment 4: Quaternion encoding ──────────────────────────────────────


def euler_to_quat(angles_deg: np.ndarray) -> np.ndarray:
    """
    Bunge Euler angles (N,3) [phi1, Phi, phi2] in degrees
    → unit quaternions (N,4) as [w, x, y, z].
    Canonical form: w >= 0  (resolves double-cover ambiguity).
    """
    from scipy.spatial.transform import Rotation
    angles_deg = np.asarray(angles_deg, dtype=np.float32)
    q_xyzw = Rotation.from_euler('ZXZ', angles_deg, degrees=True).as_quat()
    q_wxyz = np.roll(q_xyzw, shift=1, axis=-1).astype(np.float32)
    q_wxyz[q_wxyz[:, 0] < 0] *= -1
    return q_wxyz


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """
    Unit quaternions (N,4) [w, x, y, z]
    """
    from scipy.spatial.transform import Rotation
    q = np.asarray(q, dtype=np.float32)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.where(norms > 1e-8, norms, 1.0)
    q_xyzw = np.roll(q, shift=-1, axis=-1)
    angles = Rotation.from_quat(q_xyzw).as_euler(
        'ZXZ', degrees=True).astype(np.float32)
    return angles % 360.0
