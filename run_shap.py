"""
run_shap.py — GradientSHAP Analysis for SEED-IV EEG Models
============================================================
Run this script AFTER training a model to compute GradientSHAP
attributions and save per-electrode importance scores.

Usage
-----
    cd /kaggle/working/EER__XAI
    python run_shap.py \\
        --model dgcnn \\
        --checkpoint /kaggle/working/results/DGCNN/seediv_sub_wise_setting/checkpoint-bestmacro-f1 \\
        --dataset_path /kaggle/input/seed-iv-dataset \\
        --output_dir /kaggle/working/xai_results \\
        --seed 2024

Arguments
---------
    --model         : one of dgcnn | hslt | eegnet | acrnn | tsception
    --checkpoint    : full path to the saved checkpoint file
    --dataset_path  : path to SEED-IV dataset root
    --output_dir    : directory to save SHAP results (default: ./xai_results)
    --seed          : random seed — MUST match the seed used in training so
                      the test subject split is identical (default: 2024)
    --n_bg          : number of background samples for GradientSHAP (default: 100)
    --n_explain     : max number of test samples to explain (default: 300)
    --n_steps       : number of interpolation steps for GradientSHAP (default: 50)
    --batch_chunk   : SHAP batch size per forward pass (default: 50)
    --save_test_data: also save the raw test X/y arrays for local analysis

Outputs (saved to --output_dir/<model>/)
-----------------------------------------
    shap_values.npy          raw per-sample attributions (N, *input_shape)
    electrode_importance.npy per-sample electrode importance  (N, 62)
    emotion_importance.npy   mean per-emotion importance       (4, 62)
    test_data.pkl            test samples + labels (for local_analysis.py)
    shap_summary.json        runtime metadata
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# ── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.absolute()
LIBEER_SRC = SCRIPT_DIR / "LibEER"

for p in [str(LIBEER_SRC), str(SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── LibEER imports ───────────────────────────────────────────────────────────
from config.setting import Setting              # noqa: E402
from data_utils.load_data import get_data       # noqa: E402
from data_utils.split import (                  # noqa: E402
    merge_to_part, index_to_data, get_split_index
)
from data_utils.preprocess import normalize     # noqa: E402
from models.Models import Model                 # noqa: E402
from utils.utils import setup_seed              # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────
N_ELECTRODES   = 62
N_CLASSES      = 4
EMOTION_LABELS = {0: "neutral", 1: "sad", 2: "fear", 3: "happy"}

# Per-model config: which LibEER dataset + how to preprocess raw data
MODEL_CFG = {
    "dgcnn":     {"dataset": "seediv_de_lds", "only_seg": False, "sample_length": 1,   "stride": 1,   "acrnn_prep": False},
    "hslt":      {"dataset": "seediv_de_lds", "only_seg": False, "sample_length": 1,   "stride": 1,   "acrnn_prep": False},
    "eegnet":    {"dataset": "seediv_raw",    "only_seg": True,  "sample_length": 200, "stride": 200, "acrnn_prep": False},
    "tsception": {"dataset": "seediv_raw",    "only_seg": True,  "sample_length": 200, "stride": 200, "acrnn_prep": False},
    "acrnn":     {"dataset": "seediv_raw",    "only_seg": True,  "sample_length": 200, "stride": 200, "acrnn_prep": True},
}

# LibEER model class names
LIBEER_MODEL_KEY = {
    "dgcnn":     "DGCNN",
    "hslt":      "HSLT",
    "eegnet":    "EEGNet",
    "tsception": "TSception",
    "acrnn":     "ACRNN",
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_test_data(model_name: str, dataset_path: str, seed: int):
    """
    Reload the SAME test split that was used during training by using the
    identical setting + seed combination.

    Returns
    -------
    X_test : np.ndarray  raw test data before model-specific reshape
    y_test : np.ndarray  integer class labels (N,)
    channels     : int   number of EEG channels (62)
    feature_dim  : int   number of features per channel
    num_classes  : int   number of emotion classes (4)
    """
    cfg = MODEL_CFG[model_name]

    setting = Setting(
        dataset=cfg["dataset"],
        dataset_path=dataset_path,
        pass_band=[0.3, 50],
        extract_bands=None,
        time_window=1,
        overlap=0,
        sample_length=cfg["sample_length"],
        stride=cfg["stride"],
        seed=seed,
        feature_type="de_lds" if "de_lds" in cfg["dataset"] else "raw",
        only_seg=cfg["only_seg"],
        experiment_mode="subject-independent",
        normalize=False,      # LibEER ignores this; data is returned unnormalized
        save_data=False,
        split_type="train-val-test-subject-wise",
        sessions=[1, 2, 3],
        onehot=True,
    )

    print(f"  Loading {cfg['dataset']} ...")
    setup_seed(seed)
    all_data, all_label, channels, feature_dim, num_classes = get_data(setting)
    all_data, all_label = merge_to_part(all_data, all_label, setting)

    # merge_to_part returns [[subject_data, ...], ...]  — take first round
    data_i  = all_data[0]
    label_i = all_label[0]

    setup_seed(seed)   # same seed → identical subject shuffle
    tts = get_split_index(data_i, label_i, setting)

    train_idx = tts["train"][0]
    val_idx   = tts["val"][0]
    test_idx  = tts["test"][0]
    print(f"  Subject split → train: {len(train_idx)} | val: {len(val_idx)} | test: {len(test_idx)}")

    _, _, _, _, X_test, y_test = index_to_data(
        data_i, label_i, train_idx, test_idx, val_idx
    )

    # No normalization — LibEER does NOT normalize data during training
    # (setting.normalize is accepted but ignored by get_data/split).
    # Applying normalization here would cause a train/eval input mismatch.
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test)

    # Convert one-hot labels to integer class indices
    if y_test.ndim == 2:
        y_test = np.argmax(y_test, axis=1)
    y_test = y_test.astype(np.int64)

    print(f"  Test data: X={X_test.shape}  y={y_test.shape}")
    return X_test, y_test, channels, feature_dim, num_classes


# ============================================================================
# MODEL-SPECIFIC INPUT PREPARATION
# ============================================================================

def prepare_model_input(X: np.ndarray, model_name: str) -> np.ndarray:
    """
    Apply the same input reshape as the original train script.

    EEGNet    forward: x.reshape(N, 1, C, T)  → expects (N, C, T) from loader
    TSception forward: x.unsqueeze(1)          → expects (N, C, T) from loader
    ACRNN     train:   np.transpose(X,(0,2,1))[:, newaxis, :, :] → (N,1,T,C)
    DGCNN/HSLT: (N, C, bands) as-is
    """
    if model_name == "acrnn":
        # (N, 62, 200) → (N, 200, 62) → (N, 1, 200, 62)
        return np.transpose(X, (0, 2, 1))[:, np.newaxis, :, :].astype(np.float32)
    # All other models receive raw shape from the loader
    return X.astype(np.float32)


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_name: str, checkpoint_path: str,
               channels: int, feature_dim: int, num_classes: int,
               device: torch.device) -> torch.nn.Module:
    """Load LibEER model from checkpoint."""
    key = LIBEER_MODEL_KEY[model_name]

    if model_name == "dgcnn":
        m = Model[key](num_electrodes=channels, in_channels=feature_dim, num_classes=num_classes)
    elif model_name == "hslt":
        m = Model[key](num_electrodes=channels, in_channels=feature_dim, num_classes=num_classes)
    elif model_name == "eegnet":
        m = Model[key](num_electrodes=channels, datapoints=feature_dim, num_classes=num_classes)
    elif model_name == "tsception":
        m = Model[key](num_electrodes=channels, num_datapoints=feature_dim, num_classes=num_classes)
    elif model_name == "acrnn":
        # ACRNN: n_channels=electrodes, n_timepoints=timepoints
        m = Model[key](n_channels=channels, n_timepoints=feature_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    # LibEER save_state uses key 'model'
    state_dict = ckpt.get("model", ckpt.get("model_state", ckpt))
    m.load_state_dict(state_dict)
    m.eval().to(device)
    print(f"  Checkpoint loaded from {checkpoint_path}")
    return m


# ============================================================================
# ELECTRODE AGGREGATION
# ============================================================================

def to_electrode_importance(sv: np.ndarray, model_name: str) -> np.ndarray:
    """
    Collapse raw SHAP attribution to per-electrode importance (N, 62).

    Attribution shapes (same as model input shape):
      dgcnn, hslt      : (N, 62, 5)    → mean over bands
      eegnet, tsception: (N, 62, 200)  → mean over time  [EEGNet takes (N,C,T) directly]
      acrnn            : (N, 1, 200, 62) → squeeze + mean over time
    """
    sv_abs = np.abs(sv)
    if model_name in ("dgcnn", "hslt"):
        return sv_abs.mean(axis=-1)                    # (N, 62, 5) → (N, 62)
    elif model_name in ("eegnet", "tsception"):
        return sv_abs.mean(axis=-1)                    # (N, 62, 200) → (N, 62)
    elif model_name == "acrnn":
        return sv_abs.squeeze(axis=1).mean(axis=1)     # (N,1,200,62) → (N,62)
    raise ValueError(f"Unknown model: {model_name}")


# ============================================================================
# GRADIENTSHAP
# ============================================================================

def run_gradient_shap(model: torch.nn.Module, X_bg: np.ndarray,
                      X_explain: np.ndarray, y_explain: np.ndarray,
                      model_name: str, device: torch.device,
                      n_steps: int = 50, batch_chunk: int = 50) -> dict:
    """
    GradientSHAP (shap.GradientExplainer) — Expected Gradients method.

    Combines Integrated Gradients with SHAP's Shapley axioms.
    Works with any differentiable PyTorch model.

    Parameters
    ----------
    model      : trained model (eval mode)
    X_bg       : background samples, same shape as model input
    X_explain  : samples to explain, same shape as model input
    y_explain  : true class labels for X_explain (integers)
    model_name : used for electrode aggregation
    device     : torch device
    n_steps    : number of interpolation steps
    batch_chunk: samples per shap forward-pass chunk

    Returns
    -------
    dict with keys: shap_values, electrode_importance, emotion_importance
    """
    import shap

    bg_t = torch.FloatTensor(X_bg).to(device)
    xp_t = torch.FloatTensor(X_explain).to(device)

    print(f"  GradientSHAP | background: {X_bg.shape} | explain: {X_explain.shape}")
    explainer = shap.GradientExplainer(model, bg_t)

    all_sv = []
    n = len(xp_t)
    for start in range(0, n, batch_chunk):
        batch = xp_t[start: start + batch_chunk]
        sv = explainer.shap_values(batch)

        # Normalise to (chunk, *input_shape, n_classes)
        if isinstance(sv, list):
            sv_stack = np.stack(sv, axis=-1)   # list[n_cls] each (chunk,*shp) → (..., n_cls)
        else:
            sv_stack = sv
        all_sv.append(sv_stack)
        print(f"    processed {min(start + batch_chunk, n)}/{n}", end="\r")

    print()
    sv_all = np.concatenate(all_sv, axis=0)    # (N, *input_shape, n_classes)

    # Select true-class attribution for each sample
    sv_tc = np.stack(
        [sv_all[i, ..., int(y_explain[i])] for i in range(len(y_explain))],
        axis=0
    )  # (N, *input_shape)

    elec_imp = to_electrode_importance(sv_tc, model_name)   # (N, 62)

    # Per-emotion mean importance
    emotion_imp = np.zeros((N_CLASSES, N_ELECTRODES))
    for c in range(N_CLASSES):
        mask = y_explain == c
        if mask.sum() > 0:
            emotion_imp[c] = np.abs(elec_imp[mask]).mean(axis=0)

    return {
        "shap_values":          sv_tc,        # (N, *input_shape)
        "electrode_importance": elec_imp,      # (N, 62)
        "emotion_importance":   emotion_imp,   # (4, 62)
    }


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Run GradientSHAP on a trained SEED-IV model")
    p.add_argument("--model",      required=True, choices=list(MODEL_CFG),
                   help="Model name")
    p.add_argument("--checkpoint", required=True,
                   help="Path to saved checkpoint file (e.g. result/DGCNN/.../checkpoint-bestmacro-f1)")
    p.add_argument("--dataset_path", required=True,
                   help="Path to SEED-IV dataset root")
    p.add_argument("--output_dir", default="./xai_results",
                   help="Directory to save SHAP outputs")
    p.add_argument("--seed",       type=int, default=2024,
                   help="Random seed — must match training seed for identical test split")
    p.add_argument("--n_bg",       type=int, default=100,
                   help="Number of background samples for GradientSHAP")
    p.add_argument("--n_explain",  type=int, default=300,
                   help="Max number of test samples to explain")
    p.add_argument("--n_steps",    type=int, default=50,
                   help="GradientSHAP interpolation steps")
    p.add_argument("--batch_chunk", type=int, default=50,
                   help="Samples per SHAP chunk (reduce if OOM)")
    p.add_argument("--save_test_data", action="store_true",
                   help="Also save raw test X/y arrays for local_analysis.py")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"GradientSHAP Analysis — {args.model.upper()}")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Seed       : {args.seed}")
    print(f"{'='*60}")

    setup_seed(args.seed)

    # ── Load test data ───────────────────────────────────────────────────────
    print("\n[1/4] Loading test data ...")
    X_raw, y_test, channels, feature_dim, num_classes = \
        load_test_data(args.model, args.dataset_path, args.seed)

    # ── Apply model-specific input transform ─────────────────────────────────
    X_model = prepare_model_input(X_raw, args.model)
    print(f"  Model input shape: {X_model.shape}")

    # ── Load model ───────────────────────────────────────────────────────────
    print("\n[2/4] Loading model ...")
    model = load_model(
        args.model, args.checkpoint, channels, feature_dim, num_classes, device
    )

    # Quick accuracy check on test set
    model.eval()
    with torch.no_grad():
        all_preds = []
        bs = 128
        for i in range(0, len(X_model), bs):
            xb = torch.FloatTensor(X_model[i:i+bs]).to(device)
            out = model(xb)
            all_preds.append(out.argmax(1).cpu().numpy())
    preds = np.concatenate(all_preds)
    acc   = (preds == y_test[:len(preds)]).mean()
    print(f"  Test accuracy (reconstructed split): {acc:.4f}")

    # ── Sample background and explain sets ───────────────────────────────────
    rng     = np.random.default_rng(args.seed)
    n_total = len(X_model)
    n_xpl   = min(args.n_explain, n_total)
    n_bg    = min(args.n_bg, n_total)

    xpl_idx  = rng.choice(n_total, size=n_xpl,  replace=False)
    bg_idx   = rng.choice(n_total, size=n_bg,   replace=False)

    X_explain = X_model[xpl_idx]
    y_explain = y_test[xpl_idx]
    X_bg      = X_model[bg_idx]

    # ── Run GradientSHAP ─────────────────────────────────────────────────────
    print("\n[3/4] Running GradientSHAP ...")
    results = run_gradient_shap(
        model, X_bg, X_explain, y_explain,
        model_name=args.model, device=device,
        n_steps=args.n_steps, batch_chunk=args.batch_chunk,
    )

    # ── Save results ─────────────────────────────────────────────────────────
    print("\n[4/4] Saving results ...")
    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "shap_values.npy",          results["shap_values"])
    np.save(out_dir / "electrode_importance.npy", results["electrode_importance"])
    np.save(out_dir / "emotion_importance.npy",   results["emotion_importance"])

    if args.save_test_data:
        data_dir = Path(args.output_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        tag = "de" if MODEL_CFG[args.model]["dataset"] == "seediv_de_lds" else "raw"
        pkl_path = data_dir / f"test_data_{args.model}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"X": X_raw, "y": y_test, "model": args.model}, f)
        print(f"  Test data saved: {pkl_path}")

    # Metadata summary
    summary = {
        "model":       args.model,
        "checkpoint":  str(args.checkpoint),
        "seed":        args.seed,
        "test_acc":    float(acc),
        "n_explained": int(n_xpl),
        "n_bg":        int(n_bg),
        "device":      str(device),
        "shap_shape":  list(results["shap_values"].shape),
    }
    with open(out_dir / "shap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved to {out_dir}/")
    print(f"  electrode_importance : {results['electrode_importance'].shape}")
    print(f"  emotion_importance   : {results['emotion_importance'].shape}")
    print(f"  Test accuracy        : {acc:.4f}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
