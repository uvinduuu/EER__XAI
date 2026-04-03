"""
Local Analysis Pipeline — SEED-IV XAI Results
===============================================
Run this script AFTER downloading and unzipping seediv_xai_results.zip from Kaggle.

It loads saved SHAP attributions and model weights, runs the full evaluation
metrics suite, and generates publication-quality figures.

Usage
-----
    python notebooks/local_analysis.py --results_dir ./results_download

Directory structure expected
----------------------------
results_download/
  checkpoints/<model>/best_model.pth
  xai/<model>/shap_values.npy
  xai/<model>/electrode_importance.npy
  xai/<model>/emotion_importance.npy
  data/test_data_de.pkl
  data/test_data_raw.pkl
  metrics/model_accuracy.json
  metrics/model_ranking.json
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import torch

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.absolute()
REPO_ROOT   = SCRIPT_DIR.parent
LIBEER_SRC  = REPO_ROOT / "LibEER"
EVAL_SRC    = REPO_ROOT / "EvaluationMetrics"

for p in [str(LIBEER_SRC), str(REPO_ROOT), str(EVAL_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from EvaluationMetrics.pytorch_eval import XAIEvaluator   # noqa: E402
from models.Models import Model                             # noqa: E402
from data_utils.preprocess import normalize                 # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────
EMOTION_LABELS   = {0: "neutral", 1: "sad", 2: "fear", 3: "happy"}
EMOTION_COLORS   = {"neutral": "#6baed6", "sad": "#2171b5",
                    "fear":    "#fd8d3c", "happy": "#e31a1c"}
N_ELECTRODES     = 62
MODELS_DE        = ["dgcnn", "hslt"]
MODELS_RAW       = ["eegnet", "acrnn", "tsception"]
ALL_MODELS       = MODELS_DE + MODELS_RAW
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SEED-IV 62-channel names (10-20 system)
CHANNEL_NAMES = [
    "FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8",
    "FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8",
    "T7","C5","C3","C1","CZ","C2","C4","C6","T8",
    "TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8",
    "P7","P5","P3","P1","PZ","P2","P4","P6","P8",
    "PO7","PO5","PO3","POZ","PO4","PO6","PO8",
    "CB1","O1","OZ","O2","CB2",
]

BRAIN_REGIONS = {
    "Prefrontal":    ["FP1","FPZ","FP2","AF3","AF4"],
    "Frontal":       ["F7","F5","F3","F1","FZ","F2","F4","F6","F8"],
    "Left Temporal": ["FT7","FC5","FC3","T7","C5","C1"],
    "Right Temporal":["FT8","FC4","FC6","T8","C2","C6","CP6"],
    "Central":       ["FC1","C3","CZ","FCZ","FC2","C4"],
    "Left Parietal": ["TP7","CP5","CP3","P7","P5","P3","P1","PO3"],
    "Parietal":      ["CP1","CP2","CPZ","PZ"],
    "Right Parietal":["TP8","CP4","P8","P6","P2","P4","PO4"],
    "Occipital":     ["PO7","PO5","POZ","PO6","PO8","CB1","O1","O2","OZ","CB2"],
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_results(results_dir: str):
    """
    Load XAI results from the Kaggle output directory.

    Actual structure produced by run_shap.py:
      <results_dir>/
        checkpoints/<model>/checkpoint-bestmacro-f1
        xai_results/<model>/{shap_values,electrode_importance,emotion_importance}.npy
        xai_results/<model>/shap_summary.json
        xai_results/data/test_data_<model>.pkl
    """
    rd = Path(results_dir)
    xai_root = rd / "xai_results"

    # Per-model test data and XAI arrays
    test_data = {}   # model_name -> {"X": ..., "y": ...}
    xai       = {}   # model_name -> {shap_values, electrode_importance, emotion_importance, test_acc}
    summaries = {}   # model_name -> shap_summary dict

    for model_name in ALL_MODELS:
        xai_dir = xai_root / model_name
        if not xai_dir.exists():
            print(f"  WARNING: no XAI data found for {model_name}, skipping.")
            continue

        # Load arrays
        xai[model_name] = {
            "shap_values":          np.load(xai_dir / "shap_values.npy"),
            "electrode_importance": np.load(xai_dir / "electrode_importance.npy"),
            "emotion_importance":   np.load(xai_dir / "emotion_importance.npy"),
        }

        # Load summary (contains test_acc)
        summary_path = xai_dir / "shap_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summaries[model_name] = json.load(f)
            xai[model_name]["test_acc"] = summaries[model_name].get("test_acc", None)

        # Load saved test data for this model
        pkl_path = xai_root / "data" / f"test_data_{model_name}.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                test_data[model_name] = pickle.load(f)

    return test_data, xai, summaries


def load_model(model_name: str, results_dir: str,
               channels: int, feature_dim: int, num_classes: int = 4):
    """Reconstruct and load a trained model from LibEER checkpoint."""
    # LibEER saves checkpoints as 'checkpoint-best<metric>'
    ckpt_path = Path(results_dir) / "checkpoints" / model_name / "checkpoint-bestmacro-f1"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if model_name == "dgcnn":
        m = Model["DGCNN"](num_electrodes=channels, in_channels=feature_dim, num_classes=num_classes)
    elif model_name == "hslt":
        m = Model["HSLT"](num_electrodes=channels, in_channels=feature_dim, num_classes=num_classes)
    elif model_name == "eegnet":
        m = Model["EEGNet"](num_electrodes=channels, datapoints=feature_dim, num_classes=num_classes)
    elif model_name == "acrnn":
        m = Model["ACRNN"](n_channels=channels, n_timepoints=feature_dim, num_classes=num_classes)
    elif model_name == "tsception":
        m = Model["TSception"](num_electrodes=channels, num_datapoints=feature_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    # LibEER save_state() stores weights under key 'model'
    state_dict = ckpt.get("model", ckpt.get("model_state", ckpt))
    m.load_state_dict(state_dict)
    m.eval().to(DEVICE)
    return m


# ============================================================================
# SHAP FUNCTION FACTORY  (used for eval metrics)
# ============================================================================

def make_shap_fn(model, model_name: str, X_bg: np.ndarray):
    """
    Returns a callable shap_fn(X) → (N, 62) electrode importance.
    Uses GradientSHAP (same method as training).
    """
    import shap as shap_lib

    bg_tensor = torch.FloatTensor(X_bg).to(DEVICE)
    explainer  = shap_lib.GradientExplainer(model, bg_tensor)

    def shap_fn(X: np.ndarray) -> np.ndarray:
        x_t = torch.FloatTensor(X).to(DEVICE)
        sv  = explainer.shap_values(x_t)              # list[n_classes] or array
        if isinstance(sv, list):
            sv_stack = np.stack(sv, axis=-1)           # (N, *shape, n_classes)
        else:
            sv_stack = sv
        # Use mean absolute attribution over classes as the electrode signal
        sv_mean = np.abs(sv_stack).mean(axis=-1)       # (N, *input_shape)

        # Aggregate to electrode level
        sv_abs = sv_mean
        if model_name in ("dgcnn", "hslt"):
            return sv_abs.mean(axis=-1)                # (N, 62)
        elif model_name in ("eegnet", "tsception"):
            return sv_abs.squeeze(1).mean(axis=-1)     # (N, 62)
        elif model_name == "acrnn":
            return sv_abs.squeeze(1).mean(axis=1)      # (N, 62)

    return shap_fn


# ============================================================================
# FIGURE 1 — ELECTRODE IMPORTANCE BAR CHART (PER MODEL)
# ============================================================================

def plot_electrode_importance(electrode_importance: np.ndarray, model_name: str,
                               output_dir: str, top_k: int = 15):
    """Bar chart of mean electrode importance across test set."""
    sorted_idx    = np.argsort(electrode_importance)[::-1]
    sorted_scores = electrode_importance[sorted_idx]
    sorted_names  = [CHANNEL_NAMES[i] for i in sorted_idx]

    colors = ["#e41a1c" if i < top_k else "#4292c6" for i in range(N_ELECTRODES)]

    fig, ax = plt.subplots(figsize=(18, 4))
    bars = ax.bar(range(N_ELECTRODES), sorted_scores, color=colors, width=0.8)
    ax.set_xticks(range(N_ELECTRODES))
    ax.set_xticklabels(sorted_names, rotation=90, fontsize=7)
    ax.set_ylabel("Mean |SHAP|", fontsize=11)
    ax.set_title(f"{model_name.upper()} — Electrode Importance (top-{top_k} highlighted)",
                 fontsize=13, fontweight="bold")
    ax.axvline(top_k - 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)

    top_patch = mpatches.Patch(color="#e41a1c", label=f"Top-{top_k} electrodes")
    ax.legend(handles=[top_patch], fontsize=9)

    fig.tight_layout()
    out = os.path.join(output_dir, f"{model_name}_electrode_importance.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# FIGURE 2 — EMOTION-SPECIFIC TOPOGRAPHIC MAP (simplified 2D scatter)
# ============================================================================

# Approximate 2D positions for 62 SEED electrodes (normalized to [-1, 1])
_POSITIONS_2D = {
    "FP1":(-0.22,0.95),"FPZ":(0,0.98),"FP2":(0.22,0.95),
    "AF3":(-0.32,0.85),"AF4":(0.32,0.85),
    "F7":(-0.70,0.70),"F5":(-0.54,0.72),"F3":(-0.38,0.73),"F1":(-0.19,0.74),
    "FZ":(0,0.75),"F2":(0.19,0.74),"F4":(0.38,0.73),"F6":(0.54,0.72),"F8":(0.70,0.70),
    "FT7":(-0.82,0.50),"FC5":(-0.62,0.52),"FC3":(-0.42,0.53),"FC1":(-0.22,0.54),
    "FCZ":(0,0.55),"FC2":(0.22,0.54),"FC4":(0.42,0.53),"FC6":(0.62,0.52),"FT8":(0.82,0.50),
    "T7":(-0.95,0.10),"C5":(-0.72,0.15),"C3":(-0.50,0.18),"C1":(-0.25,0.20),
    "CZ":(0,0.20),"C2":(0.25,0.20),"C4":(0.50,0.18),"C6":(0.72,0.15),"T8":(0.95,0.10),
    "TP7":(-0.90,-0.25),"CP5":(-0.68,-0.20),"CP3":(-0.46,-0.18),"CP1":(-0.23,-0.16),
    "CPZ":(0,-0.15),"CP2":(0.23,-0.16),"CP4":(0.46,-0.18),"CP6":(0.68,-0.20),"TP8":(0.90,-0.25),
    "P7":(-0.80,-0.55),"P5":(-0.60,-0.52),"P3":(-0.40,-0.50),"P1":(-0.20,-0.48),
    "PZ":(0,-0.47),"P2":(0.20,-0.48),"P4":(0.40,-0.50),"P6":(0.60,-0.52),"P8":(0.80,-0.55),
    "PO7":(-0.65,-0.75),"PO5":(-0.44,-0.73),"PO3":(-0.22,-0.70),
    "POZ":(0,-0.70),"PO4":(0.22,-0.70),"PO6":(0.44,-0.73),"PO8":(0.65,-0.75),
    "CB1":(-0.35,-0.92),"O1":(-0.18,-0.92),"OZ":(0,-0.94),"O2":(0.18,-0.92),"CB2":(0.35,-0.92),
}


def plot_emotion_topomap(emotion_importance: np.ndarray, model_name: str, output_dir: str):
    """
    2×2 topographic maps showing electrode importance per emotion class.
    Uses scatter-plot proxy since mne/cartopy not required.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for c in range(4):
        ax    = axes[c]
        imp   = emotion_importance[c]   # (62,)
        vmax  = np.percentile(imp, 95)
        norm_imp = imp / (vmax + 1e-12)

        xs, ys, cs_vals = [], [], []
        for ch, name in enumerate(CHANNEL_NAMES):
            pos = _POSITIONS_2D.get(name)
            if pos is None:
                continue
            xs.append(pos[0])
            ys.append(pos[1])
            cs_vals.append(norm_imp[ch])

        # Draw head circle
        head_circle = plt.Circle((0, 0), 1.0, fill=False, color="black", linewidth=1.5)
        ax.add_patch(head_circle)
        # Nose
        ax.plot([0, 0], [1.0, 1.12], "k-", linewidth=1.5)

        sc = ax.scatter(xs, ys, c=cs_vals, cmap="RdYlGn", vmin=0, vmax=1,
                        s=200, edgecolors="gray", linewidths=0.4, zorder=3)
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="|SHAP|")

        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.15, 1.25)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{EMOTION_LABELS[c].capitalize()}", fontsize=13, fontweight="bold",
                     color=list(EMOTION_COLORS.values())[c])

    fig.suptitle(f"{model_name.upper()} — Per-Emotion Electrode Importance",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    out = os.path.join(output_dir, f"{model_name}_emotion_topomap.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# FIGURE 3 — BRAIN REGION IMPORTANCE (all models side-by-side)
# ============================================================================

def compute_region_importance(electrode_importance: np.ndarray) -> dict:
    region_imp = {}
    for region, ch_names in BRAIN_REGIONS.items():
        ch_idx = [i for i, n in enumerate(CHANNEL_NAMES) if n in ch_names]
        region_imp[region] = electrode_importance[ch_idx].mean() if ch_idx else 0.0
    return region_imp


def plot_region_comparison(all_xai: dict, output_dir: str):
    """Grouped bar chart of brain-region importance for all models."""
    regions    = list(BRAIN_REGIONS.keys())
    model_list = [m for m in ALL_MODELS if m in all_xai]
    n_models   = len(model_list)
    n_regions  = len(regions)

    x    = np.arange(n_regions)
    w    = 0.8 / n_models
    cmap = plt.cm.get_cmap("tab10", n_models)

    fig, ax = plt.subplots(figsize=(16, 5))
    for mi, model_name in enumerate(model_list):
        elec_imp = all_xai[model_name]["electrode_importance"].mean(axis=0)  # (62,)
        reg_imp  = compute_region_importance(elec_imp)
        vals     = [reg_imp[r] for r in regions]
        # normalise per model so bars are comparable
        vmax = max(vals) + 1e-12
        vals = [v / vmax for v in vals]
        ax.bar(x + mi * w, vals, width=w, label=model_name.upper(), color=cmap(mi), alpha=0.85)

    ax.set_xticks(x + w * (n_models - 1) / 2)
    ax.set_xticklabels(regions, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Normalised mean |SHAP| (per model)", fontsize=10)
    ax.set_title("Brain Region Importance Comparison — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(output_dir, "region_comparison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# FIGURE 4 — TOP-10 ELECTRODE HEATMAP (cross-model agreement)
# ============================================================================

def plot_top10_heatmap(all_xai: dict, output_dir: str, k: int = 10):
    """
    Heatmap of top-k electrode importance per model.
    Rows = models, columns = electrodes, values = normalised importance.
    """
    model_list = [m for m in ALL_MODELS if m in all_xai]
    mat        = np.zeros((len(model_list), N_ELECTRODES))

    for mi, model_name in enumerate(model_list):
        elec_imp      = all_xai[model_name]["electrode_importance"].mean(axis=0)
        mat[mi]       = elec_imp / (elec_imp.max() + 1e-12)

    # Sort columns by mean importance
    col_order = np.argsort(mat.mean(axis=0))[::-1][:30]   # top-30 for visibility
    mat_sub   = mat[:, col_order]
    ch_labels = [CHANNEL_NAMES[i] for i in col_order]

    fig, ax = plt.subplots(figsize=(20, 3.5))
    im = ax.imshow(mat_sub, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(ch_labels)))
    ax.set_xticklabels(ch_labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(model_list)))
    ax.set_yticklabels([m.upper() for m in model_list], fontsize=10)
    plt.colorbar(im, ax=ax, label="Normalised importance")
    ax.set_title(f"Top-30 Electrode Importance Heatmap (all models)", fontsize=12, fontweight="bold")

    fig.tight_layout()
    out = os.path.join(output_dir, "electrode_heatmap.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# FIGURE 5 — EVALUATION METRICS RADAR / BAR CHART
# ============================================================================

def plot_eval_metrics(eval_results: dict, output_dir: str, k: int = 10):
    """Bar chart summary of fidelity, sensitivity, and consistency per model."""
    model_list = list(eval_results.keys())
    metrics_to_plot = {
        "Comprehensiveness": lambda r: r["fidelity"][k]["comprehensiveness"],
        "Sufficiency":       lambda r: r["fidelity"][k]["sufficiency"],
        "Consistency":       lambda r: r["consistency"]["mean_spearman"],
        "Stability\n(1−MAD)": lambda r: max(0, 1 - r["sensitivity"]["mean_absolute_deviation"]),
    }

    n_m    = len(metrics_to_plot)
    x      = np.arange(len(model_list))
    w      = 0.8 / n_m
    cmap   = plt.cm.get_cmap("Set2", n_m)

    fig, ax = plt.subplots(figsize=(12, 5))
    for mi, (metric_name, fn) in enumerate(metrics_to_plot.items()):
        vals = []
        for model_name in model_list:
            try:
                vals.append(fn(eval_results[model_name]))
            except Exception:
                vals.append(0.0)
        ax.bar(x + mi * w, vals, width=w, label=metric_name, color=cmap(mi), alpha=0.85)

    ax.set_xticks(x + w * (n_m - 1) / 2)
    ax.set_xticklabels([m.upper() for m in model_list], fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"XAI Evaluation Metrics (k={k})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(output_dir, "eval_metrics_comparison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True,
                   help="Path to the unzipped Kaggle results folder")
    p.add_argument("--figures_dir", default=None,
                   help="Where to save figures (default: results_dir/figures)")
    p.add_argument("--k", type=int, default=10,
                   help="Top-k electrodes to use in fidelity evaluation")
    p.add_argument("--n_eval_samples", type=int, default=150,
                   help="Number of test samples to use for sensitivity/consistency")
    p.add_argument("--skip_eval", action="store_true",
                   help="Skip slow evaluation metrics, only generate figures")
    return p.parse_args()


def main():
    args = parse_args()

    figures_dir = args.figures_dir or os.path.join(args.results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Figures will be saved to: {figures_dir}")

    # ── Load results ────────────────────────────────────────────────────────
    print("\nLoading results ...")
    test_data, all_xai, summaries = load_results(args.results_dir)

    # Helper: get (X, y) for a model — NO normalization (LibEER trains without it)
    def get_X_for_model(model_name):
        if model_name not in test_data:
            return None, None
        td  = test_data[model_name]
        X   = td["X"].astype(np.float32)
        y   = td["y"]
        y   = y if y.ndim == 1 else np.argmax(y, axis=1)
        # ACRNN needs (N,1,T,C) — saved as (N,C,T) → transpose + expand
        if model_name == "acrnn":
            X = np.transpose(X, (0, 2, 1))[:, np.newaxis, :, :]
        return X, y

    # ── Print accuracy table ────────────────────────────────────────────────
    print("\n" + "="*50)
    print(f"{'Model':<12} {'Test Acc':>10}")
    print("-"*50)
    for model_name in ALL_MODELS:
        if model_name in summaries:
            acc = summaries[model_name].get("test_acc", float("nan"))
            print(f"  {model_name:<12} {acc:>9.4f}")
    print("="*50)

    # ── Generate per-model figures ──────────────────────────────────────────
    print("\nGenerating figures ...")
    for model_name in ALL_MODELS:
        if model_name not in all_xai:
            continue

        elec_imp     = all_xai[model_name]["electrode_importance"]  # (N, 62) or (62,)
        emotion_imp  = all_xai[model_name]["emotion_importance"]    # (4, 62)

        # Mean over test samples if per-sample
        if elec_imp.ndim == 2:
            elec_imp_mean = elec_imp.mean(axis=0)
        else:
            elec_imp_mean = elec_imp

        plot_electrode_importance(elec_imp_mean, model_name, figures_dir)
        plot_emotion_topomap(emotion_imp, model_name, figures_dir)

    plot_region_comparison(all_xai, figures_dir)
    plot_top10_heatmap(all_xai, figures_dir)

    # ── Evaluation metrics (skip if --skip_eval) ────────────────────────────
    if args.skip_eval:
        print("\nSkipping evaluation metrics (--skip_eval).")
        return

    import shap as shap_lib   # noqa: F401

    eval_results = {}
    for model_name in ALL_MODELS:
        if model_name not in all_xai:
            continue

        print(f"\nRunning evaluation for {model_name.upper()} ...")
        X_m, y_m = get_X_for_model(model_name)

        # Load model
        ch = 62
        # ACRNN input after ACRNN transform: (N, 1, T, C) → shape[-1]=C=62, shape[2]=T
        # EEGNet/TSception: (N, C, T) → shape[-1]=T  |  DE: (N, 62, 5) → shape[-1]=5
        if model_name in MODELS_DE:
            fd = 5
        elif model_name == "acrnn":
            fd = X_m.shape[2]   # timepoints dimension after (N,1,T,C) reshape
        else:
            fd = X_m.shape[-1]  # 200 timepoints for EEGNet/TSception
        try:
            model = load_model(model_name, args.results_dir, ch, fd, num_classes=4)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        # Sample for evaluation
        rng = np.random.default_rng(0)
        n   = min(args.n_eval_samples, len(X_m))
        idx = rng.choice(len(X_m), n, replace=False)
        X_sub = X_m[idx]
        y_sub = y_m[idx]

        # Background for SHAP
        n_bg   = min(50, len(X_sub))
        bg_idx = rng.choice(n, n_bg, replace=False)
        X_bg   = X_sub[bg_idx]

        shap_fn = make_shap_fn(model, model_name, X_bg)

        elec_imp_mean = all_xai[model_name]["electrode_importance"]
        if elec_imp_mean.ndim == 2:
            elec_imp_mean = elec_imp_mean.mean(axis=0)

        evaluator    = XAIEvaluator(model, DEVICE, model_name)
        eval_results[model_name] = evaluator.evaluate_all(
            X_sub, y_sub, elec_imp_mean, shap_fn, k_values=[5, 10, 15, 20]
        )

    # Save evaluation results
    if eval_results:
        import json as _json

        def _safe(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {str(k): _safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_safe(v) for v in obj]
            return obj

        out_path = os.path.join(figures_dir, "evaluation_results.json")
        with open(out_path, "w") as f:
            _json.dump(_safe(eval_results), f, indent=2)
        print(f"\nEvaluation results saved to {out_path}")

        plot_eval_metrics(eval_results, figures_dir, k=args.k)

    print("\nLocal analysis complete.")


if __name__ == "__main__":
    main()
