"""
PyTorch-Native XAI Evaluation Metrics
=======================================
Wraps and extends the existing EvaluationMetrics/ classes for use with
PyTorch deep learning models (no sklearn API required).

Metrics
-------
1. Comprehensiveness  : accuracy drop when top-k important electrodes are masked
2. Sufficiency        : accuracy using ONLY top-k important electrodes
3. Sensitivity        : explanation stability under small input perturbation
4. Consistency        : Spearman rank correlation of explanations across similar samples

Usage
-----
    from EvaluationMetrics.pytorch_eval import XAIEvaluator

    evaluator = XAIEvaluator(model, device)
    results = evaluator.evaluate_all(
        X_test, y_test, electrode_importance, model_name
    )
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from typing import Optional


# ============================================================================
# HELPERS
# ============================================================================

def _predict_proba(model: nn.Module, X: np.ndarray, device: torch.device,
                   batch_size: int = 64) -> np.ndarray:
    """Run model inference and return softmax probabilities (N, n_classes)."""
    model.eval()
    probs_list = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[start: start + batch_size]).to(device)
            logits = model(batch)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_list.append(probs)
    return np.concatenate(probs_list, axis=0)


def _electrode_mask_de(X: np.ndarray, top_k_idx: np.ndarray,
                        mode: str = "remove") -> np.ndarray:
    """
    Mask electrodes in DE-feature inputs (N, 62, 5).

    mode='remove'  → zero out the top-k electrodes  (comprehensiveness)
    mode='keep'    → zero out all EXCEPT top-k       (sufficiency)
    """
    X_copy = X.copy()
    if mode == "remove":
        X_copy[:, top_k_idx, :] = 0.0
    else:
        mask = np.ones(X.shape[1], dtype=bool)
        mask[top_k_idx] = False
        X_copy[:, mask, :] = 0.0
    return X_copy


def _electrode_mask_raw_eegnet_ts(X: np.ndarray, top_k_idx: np.ndarray,
                                   mode: str = "remove") -> np.ndarray:
    """Mask electrodes in raw EEG inputs (N, 1, 62, 200)."""
    X_copy = X.copy()
    if mode == "remove":
        X_copy[:, :, top_k_idx, :] = 0.0
    else:
        mask = np.ones(X.shape[2], dtype=bool)
        mask[top_k_idx] = False
        X_copy[:, :, mask, :] = 0.0
    return X_copy


def _electrode_mask_acrnn(X: np.ndarray, top_k_idx: np.ndarray,
                           mode: str = "remove") -> np.ndarray:
    """Mask electrodes in ACRNN inputs (N, 1, 200, 62)."""
    X_copy = X.copy()
    if mode == "remove":
        X_copy[:, :, :, top_k_idx] = 0.0
    else:
        mask = np.ones(X.shape[3], dtype=bool)
        mask[top_k_idx] = False
        X_copy[:, :, :, mask] = 0.0
    return X_copy


def mask_electrodes(X: np.ndarray, top_k_idx: np.ndarray, model_name: str,
                    mode: str = "remove") -> np.ndarray:
    """Dispatch to the correct mask helper based on model name."""
    if model_name in ("dgcnn", "hslt"):
        return _electrode_mask_de(X, top_k_idx, mode)
    elif model_name in ("eegnet", "tsception"):
        return _electrode_mask_raw_eegnet_ts(X, top_k_idx, mode)
    elif model_name == "acrnn":
        return _electrode_mask_acrnn(X, top_k_idx, mode)
    raise ValueError(f"Unknown model: {model_name}")


def _add_noise(X: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise to input array."""
    return X + rng.normal(0, std, size=X.shape).astype(X.dtype)


# ============================================================================
# MAIN EVALUATOR CLASS
# ============================================================================

class XAIEvaluator:
    """
    Evaluates GradientSHAP explanations for PyTorch EEG models.

    Parameters
    ----------
    model      : trained nn.Module (must be in eval mode after training)
    device     : torch.device
    model_name : one of "dgcnn", "hslt", "eegnet", "acrnn", "tsception"
    """

    def __init__(self, model: nn.Module, device: torch.device, model_name: str):
        self.model      = model.eval()
        self.device     = device
        self.model_name = model_name

    # ------------------------------------------------------------------
    # 1. COMPREHENSIVENESS
    # ------------------------------------------------------------------

    def comprehensiveness(self, X: np.ndarray, y: np.ndarray,
                          electrode_importance: np.ndarray,
                          k_values: list = (5, 10, 15, 20)) -> dict:
        """
        Comprehensiveness = accuracy_full − accuracy_top_k_removed.

        Higher is better: removing the most important electrodes should hurt
        performance the most if the explanation is faithful.

        Parameters
        ----------
        X                    : input array (N, *input_shape)
        y                    : integer class labels (N,)
        electrode_importance : mean importance per electrode (62,)
        k_values             : list of k values to evaluate

        Returns dict  {k: comprehensiveness_score}
        """
        # Baseline accuracy with full input
        probs_full  = _predict_proba(self.model, X, self.device)
        acc_full    = (probs_full.argmax(1) == y).mean()

        top_idx_all = np.argsort(electrode_importance)[::-1]   # sorted desc by importance

        results = {}
        for k in k_values:
            top_k_idx  = top_idx_all[:k]
            X_removed  = mask_electrodes(X, top_k_idx, self.model_name, mode="remove")
            probs_mask  = _predict_proba(self.model, X_removed, self.device)
            acc_removed = (probs_mask.argmax(1) == y).mean()
            results[k]  = float(acc_full - acc_removed)   # positive = explanation is useful

        print(f"  Comprehensiveness (acc_full={acc_full:.3f}): {results}")
        return results

    # ------------------------------------------------------------------
    # 2. SUFFICIENCY
    # ------------------------------------------------------------------

    def sufficiency(self, X: np.ndarray, y: np.ndarray,
                    electrode_importance: np.ndarray,
                    k_values: list = (5, 10, 15, 20)) -> dict:
        """
        Sufficiency = accuracy using ONLY the top-k important electrodes.

        Higher is better: if explanation is faithful, the top-k electrodes
        alone should be sufficient to maintain high accuracy.

        Returns dict  {k: sufficiency_score}
        """
        top_idx_all = np.argsort(electrode_importance)[::-1]

        results = {}
        for k in k_values:
            top_k_idx = top_idx_all[:k]
            X_kept    = mask_electrodes(X, top_k_idx, self.model_name, mode="keep")
            probs     = _predict_proba(self.model, X_kept, self.device)
            acc       = (probs.argmax(1) == y).mean()
            results[k] = float(acc)

        print(f"  Sufficiency: {results}")
        return results

    # ------------------------------------------------------------------
    # 3. SENSITIVITY (adapted from EvaluationMetrics/Sensitivity.py)
    # ------------------------------------------------------------------

    def sensitivity(self, X: np.ndarray,
                    shap_fn,                   # callable: X → (N, 62) electrode importance
                    perturbation_std: float = 0.01,
                    n_perturbations: int    = 30,
                    n_samples: int          = 100,
                    seed: int               = 42) -> dict:
        """
        Sensitivity = stability of electrode-level SHAP explanation
        under small Gaussian noise on the input.

        Metrics returned (lower MAD / higher cosine = more stable):
          - mean_absolute_deviation  (MAD)  : mean |orig − perturbed| importance
          - cosine_similarity               : mean cos(orig, perturbed)
          - std_deviation                   : std of MADs across samples

        shap_fn : a callable that takes X (numpy, same shape as model input)
                  and returns electrode_importance (N, 62).
                  Example:
                      shap_fn = lambda x: evaluator._quick_shap(x)
        """
        rng = np.random.default_rng(seed)
        n_samples = min(n_samples, len(X))
        idx_sample = rng.choice(len(X), size=n_samples, replace=False)
        X_sub = X[idx_sample]

        mad_list   = []
        cos_list   = []

        for i in range(n_samples):
            xi    = X_sub[i: i+1]                 # (1, *shape)
            orig  = shap_fn(xi).squeeze()          # (62,)

            pert_imps = []
            for _ in range(n_perturbations):
                xi_p = _add_noise(xi, perturbation_std, rng)
                pi   = shap_fn(xi_p).squeeze()     # (62,)
                pert_imps.append(pi)

            pert_arr = np.stack(pert_imps, axis=0) # (n_pert, 62)
            mad      = np.abs(pert_arr - orig).mean(axis=1).mean()
            cos_sims = []
            for pv in pert_imps:
                no = np.linalg.norm(orig)
                np_ = np.linalg.norm(pv)
                if no < 1e-12 and np_ < 1e-12:
                    cos_sims.append(1.0)
                elif no < 1e-12 or np_ < 1e-12:
                    cos_sims.append(0.0)
                else:
                    cos_sims.append(float(np.dot(orig, pv) / (no * np_)))

            mad_list.append(mad)
            cos_list.append(np.mean(cos_sims))

        results = {
            "mean_absolute_deviation": float(np.mean(mad_list)),
            "cosine_similarity":       float(np.mean(cos_list)),
            "std_deviation":           float(np.std(mad_list)),
        }
        print(f"  Sensitivity: MAD={results['mean_absolute_deviation']:.6f}  "
              f"cos={results['cosine_similarity']:.4f}")
        return results

    # ------------------------------------------------------------------
    # 4. CONSISTENCY (rank correlation across similar samples)
    # ------------------------------------------------------------------

    def consistency(self, X: np.ndarray, y: np.ndarray,
                    shap_fn,
                    n_samples: int = 50,
                    seed: int      = 42) -> dict:
        """
        Consistency = mean Spearman rank correlation between electrode
        importance explanations for same-class sample pairs.

        Same-class samples should produce similar electrode rankings if
        the model has learned stable emotion-specific patterns.

        Returns dict:
          - mean_spearman   : mean ρ across same-class pairs
          - per_class       : {emotion: mean ρ}
        """
        rng = np.random.default_rng(seed)
        n_classes  = len(np.unique(y))
        per_class  = {}
        all_rhos   = []

        for c in range(n_classes):
            idx_c = np.where(y == c)[0]
            if len(idx_c) < 2:
                continue
            chosen = rng.choice(idx_c, size=min(n_samples * 2, len(idx_c)), replace=False)
            imp_c  = shap_fn(X[chosen])   # (M, 62)

            # Compute pairwise Spearman correlations between consecutive pairs
            rhos_c = []
            for j in range(0, len(imp_c) - 1, 2):
                rho, _ = spearmanr(imp_c[j], imp_c[j+1])
                if not np.isnan(rho):
                    rhos_c.append(rho)

            per_class[c] = float(np.mean(rhos_c)) if rhos_c else 0.0
            all_rhos.extend(rhos_c)

        EMOTION_LABELS = {0: "neutral", 1: "sad", 2: "fear", 3: "happy"}
        per_class_named = {EMOTION_LABELS.get(k, str(k)): v for k, v in per_class.items()}

        results = {
            "mean_spearman": float(np.mean(all_rhos)) if all_rhos else 0.0,
            "per_class":     per_class_named,
        }
        print(f"  Consistency: mean_spearman={results['mean_spearman']:.4f}")
        return results

    # ------------------------------------------------------------------
    # 5. FIDELITY WRAPPER (extends EvaluationMetrics/Fidelity.py logic)
    # ------------------------------------------------------------------

    def fidelity(self, X: np.ndarray, y: np.ndarray,
                 electrode_importance: np.ndarray,
                 k_values: list = (5, 10, 15, 20)) -> dict:
        """
        Convenience method that computes both Comprehensiveness and Sufficiency
        and returns a unified fidelity score per k.

        Fidelity_k  = (comprehensiveness_k + sufficiency_k) / 2
        (based on ERASER benchmark: DeYoung et al. 2020, ACL)
        """
        comp = self.comprehensiveness(X, y, electrode_importance, k_values)
        suff = self.sufficiency(      X, y, electrode_importance, k_values)

        fidelity = {}
        for k in k_values:
            fidelity[k] = {
                "comprehensiveness": comp[k],
                "sufficiency":       suff[k],
                "fidelity":          (comp[k] + suff[k]) / 2,
            }

        print(f"  Fidelity summary:")
        for k, v in fidelity.items():
            print(f"    k={k:2d}: comp={v['comprehensiveness']:.4f}  "
                  f"suff={v['sufficiency']:.4f}  fidelity={v['fidelity']:.4f}")
        return fidelity

    # ------------------------------------------------------------------
    # EVALUATE ALL
    # ------------------------------------------------------------------

    def evaluate_all(self, X: np.ndarray, y: np.ndarray,
                     electrode_importance: np.ndarray,
                     shap_fn,
                     k_values: list = (5, 10, 15, 20)) -> dict:
        """
        Run the complete evaluation suite.

        Parameters
        ----------
        X                    : test input (N, *input_shape)
        y                    : integer labels (N,)
        electrode_importance : mean importance over test set (62,)
        shap_fn              : callable(X) → (N, 62) electrode importance
        k_values             : top-k values for masking tests

        Returns full evaluation dict.
        """
        print(f"\n{'='*60}")
        print(f"XAI Evaluation: {self.model_name.upper()}")
        print(f"{'='*60}")

        results = {
            "model":   self.model_name,
            "n_test":  len(X),
        }

        results["fidelity"]    = self.fidelity(X, y, electrode_importance, k_values)
        results["sensitivity"] = self.sensitivity(X, shap_fn)
        results["consistency"] = self.consistency(X, y, shap_fn)

        return results
