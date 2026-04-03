"""
Fidelity Evaluation for XAI Methods (SHAP & LIME)


Fidelity measures how accurately the XAI explanation represents what the model 
is ACTUALLY doing. Does the simple explanation match the complex reality?



we implement comprehensive fidelity assessment tool to measure how faithfully
XAI explanations (SHAP, LIME) represent the original model's behavior.

Key Features:
- Probability-based fidelity metrics (MAE, KL divergence, R²)
- Proper SHAP logit-to-probability conversion for LinearExplainer
- Multi-class classification support with softmax calibration
- Neighborhood-based local fidelity measurement
- Distributional fidelity via normalized cross-entropy
- Batch evaluation for large-scale experiments


Metrics Provided:
1. SHAP Neighborhood Fidelity: Linear approximation accuracy in local region
2. SHAP Distributional Fidelity: Cross-entropy based probability matching
3. LIME Neighborhood Fidelity: Weighted surrogate model accuracy

Mathematical Foundation:
- Fidelity = 1 - MAE(true_probs, surrogate_probs) for classification
- Fidelity = R² score for regression
- Cross-entropy normalized by log(n_classes) for distributional metrics


"""

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelBinarizer
import warnings
from typing import Dict




def shap_surrogate_probs_fixed(explainer, X_pert, link='softmax'):
    """
    For LinearExplainer:
    - SHAP values are in LOG-ODDS space (logits)
    - base_value is the log-odds at the mean feature values
    - We need to convert: logit → probability via sigmoid/softmax
    
    For TreeExplainer:
    - SHAP values are in probability space
    """
    try:
        sv = explainer.shap_values(X_pert, check_additivity=False)
    except TypeError:
        sv = explainer.shap_values(X_pert)
    
    base_raw = np.array(explainer.expected_value)
    
    explainer_type = type(explainer).__name__

    if isinstance(sv, list):
        n_classes = len(sv)
        sv_list = sv
    else:
        sv_arr = np.asarray(sv)
        if sv_arr.ndim == 3:
            n_classes = sv_arr.shape[2]
            sv_list = [sv_arr[:, :, c] for c in range(n_classes)]
        else:
            n_classes = 2
            sv_list = [sv_arr, -sv_arr]
    
    # Align base values
    if base_raw.ndim == 0:
        base_arr = np.repeat(float(base_raw), n_classes)
    else:
        base_arr = np.asarray(base_raw, dtype=float).ravel()
        if base_arr.size < n_classes:
            base_arr = np.pad(base_arr, (0, n_classes - base_arr.size), 
                            mode='constant', constant_values=0)
        elif base_arr.size > n_classes:
            base_arr = base_arr[:n_classes]
    
    n_samples = sv_list[0].shape[0]
    
    if explainer_type == 'LinearExplainer':
        # LinearExplainer: SHAP values are already in logit space
        # for multiclass LR, we compute logits for each class
        logits = np.zeros((n_samples, n_classes), dtype=float)
        
        for c in range(n_classes):
            # sum SHAP contributions
            logits[:, c] = base_arr[c] + sv_list[c].sum(axis=1)
        
        #  convert softmax to  logits → probabilities
        logits_stable = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits_stable)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
    else:
        # TreeExplainer: SHAP values are in margin/score space
        logits = np.zeros((n_samples, n_classes), dtype=float)
        
        for c in range(n_classes):
            logits[:, c] = base_arr[c] + sv_list[c].sum(axis=1)
        
        if link == 'softmax':
            logits_stable = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits_stable)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        elif link == 'logit':
            if n_classes == 2:
                p1 = expit(logits[:, 1])
                probs = np.column_stack([1 - p1, p1])
            else:
                probs = softmax(logits, axis=1)
        else:
            raise ValueError(f"Unknown link: {link}")
    
    return probs


def shap_neighborhood_fidelity(
    explainer, model, instance, n_perturb=50, noise_std=0.01, 
    task_type='classification', random_state=None, link='softmax'
):
    """
    Measure how well SHAP's linear approximation matches the true model
    in a local neighborhood.
    
    Fidelity = 1 - MAE(model_probs, shap_linear_approx_probs)
    """
    X0 = instance.values if hasattr(instance, "values") else np.asarray(instance)
    X0 = np.atleast_2d(X0)
    n_feat = X0.shape[1]

    if X0.shape[0] != 1:
        raise ValueError("instance must be a single row")
    if n_perturb <= 0:
        raise ValueError("n_perturb must be positive")

    # here we generate perturbed neighborhood
    rng = np.random.default_rng(random_state)
    noise = rng.normal(0, noise_std, size=(n_perturb, n_feat))
    X_pert = X0 + noise

    if task_type == 'classification':
        # true model probabilities
        Y_true = np.asarray(model.predict_proba(X_pert))
        
        # SHAP linear approximation probabilities
        Y_shap = shap_surrogate_probs_fixed(explainer, X_pert, link=link)
        
        # MAE between probability distributions
        mae = np.abs(Y_true - Y_shap).mean()
        fidelity = 1.0 - mae
        return float(np.clip(fidelity, 0, 1))
    
    else:  
        # regression
        Y_true = model.predict(X_pert)
        
        # For regression, we use SHAP values directly
        try:
            sv = explainer.shap_values(X_pert, check_additivity=False)
        except TypeError:
            sv = explainer.shap_values(X_pert)
        
        sv_arr = np.asarray(sv)
        base = float(np.asarray(explainer.expected_value).item())
        
        if sv_arr.ndim == 2:
            Y_shap = base + sv_arr.sum(axis=1)
        else:
            Y_shap = base + sv_arr.sum()
        
        r2 = r2_score(Y_true, Y_shap)
        return float(np.clip(r2, 0, 1))


def lime_neighborhood_fidelity(
    explainer, model, instance, n_perturb=50, noise_std=0.01, 
    task_type='classification', random_state=None, kernel_width=None
):
    """
    Measure how well LIME's weighted ridge surrogate matches the true model.
    """
    X0 = instance.values if hasattr(instance, "values") else np.asarray(instance)
    X0 = np.atleast_2d(X0)
    n_feat = X0.shape[1]

    # Determine kernel width
    if kernel_width is None:
        if hasattr(explainer, 'kernel_width'):
            kernel_width = explainer.kernel_width
        elif hasattr(explainer, '_LimeTabularExplainer__kernel_width'):
            kernel_width = explainer._LimeTabularExplainer__kernel_width
        else:
            kernel_width = np.sqrt(n_feat)
    
    # Generate perturbed neighborhood
    rng = np.random.default_rng(random_state)
    noise = rng.normal(0, noise_std, size=(n_perturb, n_feat))
    X_pert = X0 + noise

    # compute LIME kernel weights
    dists = np.linalg.norm(X_pert - X0, axis=1)
    weights = np.exp(-(dists ** 2) / (kernel_width ** 2))

    if task_type == 'classification':
        # true model probabilities
        Y_true = np.asarray(model.predict_proba(X_pert))
        K = Y_true.shape[1]
        
        # fit weighted ridge for each class
        Y_sur = np.zeros_like(Y_true)
        for k in range(K):
            ridge = Ridge(alpha=1.0, fit_intercept=True)
            ridge.fit(X_pert, Y_true[:, k], sample_weight=weights)
            Y_sur[:, k] = ridge.predict(X_pert)
        
        # Safe normalization
        Y_sur = np.maximum(Y_sur, 0)
        row_sums = Y_sur.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        Y_sur = Y_sur / row_sums
        
        mae = np.abs(Y_true - Y_sur).mean()
        fidelity = 1.0 - mae
        return float(np.clip(fidelity, 0, 1))
    
    else:  
        # regression
        Y_true = model.predict(X_pert)
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(X_pert, Y_true, sample_weight=weights)
        Y_sur = ridge.predict(X_pert)
        
        r2 = r2_score(Y_true, Y_sur)
        return float(np.clip(r2, 0, 1))


def shap_distributional_fidelity(
    explainer, model, instance, n_perturb=50, noise_std=0.01, 
    feature_info=None, random_state=None, link='softmax'
):
    """
    Distributional fidelity via normalized cross-entropy.
    """
    X0 = instance.values if hasattr(instance, "values") else np.asarray(instance)
    X0 = np.atleast_2d(X0)
    
    rng = np.random.default_rng(random_state)
    noise = rng.normal(0, noise_std, size=(n_perturb, X0.shape[1]))
    X_pert = X0 + noise

    P_true = np.asarray(model.predict_proba(X_pert))
    K = P_true.shape[1]

    P_sur = shap_surrogate_probs_fixed(explainer, X_pert, link=link)

    eps = 1e-12
    P_sur_clipped = np.clip(P_sur, eps, 1.0)
    CE = -np.sum(P_true * np.log(P_sur_clipped), axis=1)
    ce_mean = float(np.mean(CE))
    
    ce_max = np.log(K)
    fidelity = 1.0 - ce_mean / ce_max
    return float(np.clip(fidelity, 0, 1))


def batch_fidelity_evaluation(
    explainer_dict: Dict,  
    model_dict: Dict,      
    X_test,
    y_test,
    dataset_name: str,
    n_instances: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Batch evaluate fidelity across multiple models and explainers.
    
    Returns: DataFrame with columns [Model, Explainer, Instance, SHAP_Neighborhood, 
                                     SHAP_Distributional, LIME_Neighborhood]
    """
    results = []
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_test), min(n_instances, len(X_test)), replace=False)
    
    for model_name, model in model_dict.items():
        for i in idx:
            instance = X_test[i:i+1]
            
            # SHAP metrics
            if 'shap' in explainer_dict:
                shap_exp = explainer_dict['shap']
                try:
                    shap_neigh = shap_neighborhood_fidelity(
                        shap_exp, model, instance, 
                        n_perturb=100, random_state=random_state
                    )
                    shap_dist = shap_distributional_fidelity(
                        shap_exp, model, instance, 
                        n_perturb=100, random_state=random_state
                    )
                except Exception as e:
                    print(f"SHAP error ({model_name}, instance {i}): {e}")
                    shap_neigh, shap_dist = np.nan, np.nan
            
            # LIME metrics
            if 'lime' in explainer_dict:
                lime_exp = explainer_dict['lime']
                try:
                    lime_neigh = lime_neighborhood_fidelity(
                        lime_exp, model, instance, 
                        n_perturb=100, random_state=random_state
                    )
                except Exception as e:
                    print(f"LIME error ({model_name}, instance {i}): {e}")
                    lime_neigh = np.nan
            
            results.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Instance': i,
                'SHAP_Neighborhood': shap_neigh,
                'SHAP_Distributional': shap_dist,
                'LIME_Neighborhood': lime_neigh
            })
    
    return pd.DataFrame(results)



# def _softmax(z, axis=1):
#     """Numerically stable softmax."""
#     z = np.asarray(z)
#     z = z - np.max(z, axis=axis, keepdims=True)
#     ez = np.exp(z)
#     return ez / np.sum(ez, axis=axis, keepdims=True)



# def _generate_typed_neighborhood(X0, n_perturb, noise_std, feature_info=None, random_state=None):
#     """
#     Manifold-aware perturbations respecting feature types.
    
#     feature_info keys:
#       - numeric_idx: list of numeric feature indices
#       - clip_min, clip_max: arrays of per-feature bounds
#       - binary_idx: list of binary feature indices
#       - onehot_groups: list of lists (each group = one-hot encoding indices)
#       - binary_flip_prob (default 0.05), onehot_switch_prob (default 0.02)
#     """
#     X0 = np.atleast_2d(X0)
#     n_feat = X0.shape[1]
#     rng = np.random.default_rng(random_state)
#     X = np.repeat(X0, n_perturb, axis=0)

#     if feature_info is None:
#         X += rng.normal(0, noise_std, size=(n_perturb, n_feat))
#         return X
#     X_out = X.copy()
#     if feature_info.get('numeric_idx'):
#         idx = np.array(feature_info['numeric_idx'], dtype=int)
#         X_out[:, idx] = X[:, idx] + rng.normal(0, noise_std, size=(n_perturb, len(idx)))
#         mins = feature_info.get('clip_min')
#         maxs = feature_info.get('clip_max')
#         if mins is not None and maxs is not None:
#             mins = np.asarray(mins)
#             maxs = np.asarray(maxs)
#             X_out[:, idx] = np.clip(X_out[:, idx], mins[idx], maxs[idx])

#     # Binary flips
#     if feature_info.get('binary_idx'):
#         p = feature_info.get('binary_flip_prob', 0.05)
#         idx = np.array(feature_info['binary_idx'], dtype=int)
#         flips = rng.random(size=(n_perturb, len(idx))) < p
#         X_out[:, idx] = np.where(flips, 1 - X[:, idx], X[:, idx])

#     # One-hot switches
#     groups = feature_info.get('onehot_groups', [])
#     p_switch = feature_info.get('onehot_switch_prob', 0.02)
#     for grp in groups or []:
#         grp = np.array(grp, dtype=int)
#         if grp.size == 0:
#             continue
#         switch_mask = rng.random(n_perturb) < p_switch
#         if np.any(switch_mask):
#             new_idx = rng.integers(0, grp.size, size=np.sum(switch_mask))
#             X_out[switch_mask[:, None], grp] = 0
#             X_out[switch_mask, grp[new_idx]] = 1

#     return X_out
