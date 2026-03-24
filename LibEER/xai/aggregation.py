"""
Aggregation functions for XAI results.

Converts per-feature importance scores to:
- Per-electrode importance
- Per-brain-region importance
- Per-frequency-band importance
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .utils import (
    SEED_CHANNEL_NAMES,
    SEED_BRAIN_REGIONS,
    FREQUENCY_BANDS,
    get_electrode_to_region_map
)


def aggregate_to_electrode(
    xai_scores: np.ndarray,
    aggregation: str = 'mean',
    band_weights: Optional[List[float]] = None
) -> np.ndarray:
    """
    Aggregate XAI scores from (samples, electrodes, bands) to (electrodes,).

    Args:
        xai_scores: XAI importance scores, shape (n_samples, 62, 5) or (62, 5)
        aggregation: Aggregation method - 'mean', 'max', 'sum', 'weighted'
        band_weights: Weights for each frequency band (only used if aggregation='weighted')
                     Default weights emphasize alpha/beta for emotion: [0.1, 0.15, 0.25, 0.3, 0.2]

    Returns:
        Per-electrode importance scores, shape (62,)
    """
    # Handle different input shapes
    if xai_scores.ndim == 2:
        # Shape (electrodes, bands) - single sample
        scores = xai_scores
    elif xai_scores.ndim == 3:
        # Shape (samples, electrodes, bands) - average across samples first
        scores = np.abs(xai_scores).mean(axis=0)
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {xai_scores.shape}")

    # Aggregate across frequency bands
    if aggregation == 'mean':
        electrode_importance = scores.mean(axis=1)

    elif aggregation == 'max':
        electrode_importance = scores.max(axis=1)

    elif aggregation == 'sum':
        electrode_importance = scores.sum(axis=1)

    elif aggregation == 'weighted':
        if band_weights is None:
            # Default: emphasize alpha/beta bands for emotion
            # Order: delta, theta, alpha, beta, gamma
            band_weights = [0.1, 0.15, 0.25, 0.3, 0.2]
        band_weights = np.array(band_weights)
        electrode_importance = (scores * band_weights).sum(axis=1)

    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    return electrode_importance


def aggregate_by_region(
    electrode_scores: np.ndarray,
    aggregation: str = 'mean'
) -> Dict[str, float]:
    """
    Aggregate electrode scores to brain region level.

    Args:
        electrode_scores: Per-electrode importance, shape (62,)
        aggregation: 'mean', 'max', or 'sum'

    Returns:
        Dictionary mapping region names to importance scores
    """
    electrode_to_region = get_electrode_to_region_map()
    region_scores = {region: [] for region in SEED_BRAIN_REGIONS.keys()}

    # Collect scores for each region
    for i, electrode_name in enumerate(SEED_CHANNEL_NAMES):
        if electrode_name in electrode_to_region:
            region = electrode_to_region[electrode_name]
            region_scores[region].append(electrode_scores[i])

    # Aggregate
    result = {}
    for region, scores in region_scores.items():
        if scores:
            if aggregation == 'mean':
                result[region] = np.mean(scores)
            elif aggregation == 'max':
                result[region] = np.max(scores)
            elif aggregation == 'sum':
                result[region] = np.sum(scores)
        else:
            result[region] = 0.0

    return result


def aggregate_by_frequency_band(
    xai_scores: np.ndarray,
    aggregation: str = 'mean'
) -> Dict[str, float]:
    """
    Aggregate XAI scores to frequency band level.

    Args:
        xai_scores: XAI scores, shape (n_samples, 62, 5) or (62, 5)
        aggregation: 'mean', 'max', or 'sum'

    Returns:
        Dictionary mapping band names to importance scores
    """
    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    # Handle input shape
    if xai_scores.ndim == 3:
        scores = np.abs(xai_scores).mean(axis=0)  # (62, 5)
    else:
        scores = xai_scores

    result = {}
    for i, band in enumerate(band_names):
        band_scores = scores[:, i]
        if aggregation == 'mean':
            result[band] = float(np.mean(band_scores))
        elif aggregation == 'max':
            result[band] = float(np.max(band_scores))
        elif aggregation == 'sum':
            result[band] = float(np.sum(band_scores))

    return result


def aggregate_by_emotion(
    xai_scores: np.ndarray,
    labels: np.ndarray,
    aggregation: str = 'mean'
) -> Dict[str, np.ndarray]:
    """
    Compute per-electrode importance separately for each emotion class.

    Args:
        xai_scores: XAI scores, shape (n_samples, 62, 5) or (n_samples, 62)
        labels: Emotion labels, shape (n_samples,)
        aggregation: Aggregation method

    Returns:
        Dictionary mapping emotion names to electrode importance arrays
    """
    emotion_names = {0: 'neutral', 1: 'sad', 2: 'fear', 3: 'happy'}
    result = {}

    for emotion_idx, emotion_name in emotion_names.items():
        mask = labels == emotion_idx
        if mask.sum() > 0:
            emotion_scores = xai_scores[mask]

            # Aggregate to electrode level
            if emotion_scores.ndim == 3:
                # (n_samples, 62, 5) -> (62,)
                electrode_imp = aggregate_to_electrode(emotion_scores, aggregation)
            else:
                # (n_samples, 62) -> (62,)
                electrode_imp = np.abs(emotion_scores).mean(axis=0)

            result[emotion_name] = electrode_imp

    return result


def rank_electrodes(
    electrode_scores: np.ndarray,
    top_k: Optional[int] = None,
    return_indices: bool = False
) -> List[Tuple[str, float]]:
    """
    Rank electrodes by importance score.

    Args:
        electrode_scores: Per-electrode importance, shape (62,)
        top_k: Return only top K electrodes (None = return all)
        return_indices: If True, also return electrode indices

    Returns:
        List of (electrode_name, score) tuples, sorted by importance (descending)
    """
    ranked_indices = np.argsort(electrode_scores)[::-1]

    if top_k is not None:
        ranked_indices = ranked_indices[:top_k]

    result = []
    for idx in ranked_indices:
        if return_indices:
            result.append((idx, SEED_CHANNEL_NAMES[idx], electrode_scores[idx]))
        else:
            result.append((SEED_CHANNEL_NAMES[idx], electrode_scores[idx]))

    return result


def compute_hemisphere_asymmetry(
    electrode_scores: np.ndarray
) -> Dict[str, float]:
    """
    Compute asymmetry between left and right hemisphere.

    Important for emotion recognition as emotional processing
    shows hemispheric lateralization.

    Args:
        electrode_scores: Per-electrode importance, shape (62,)

    Returns:
        Dictionary with left, right, and asymmetry scores
    """
    # Define left and right hemisphere electrodes
    left_electrodes = [
        'FP1', 'AF3', 'F7', 'F5', 'F3', 'F1', 'FT7', 'FC5', 'FC3', 'FC1',
        'T7', 'C5', 'C3', 'C1', 'TP7', 'CP5', 'CP3', 'CP1',
        'P7', 'P5', 'P3', 'P1', 'PO7', 'PO5', 'PO3', 'CB1', 'O1'
    ]
    right_electrodes = [
        'FP2', 'AF4', 'F8', 'F6', 'F4', 'F2', 'FT8', 'FC6', 'FC4', 'FC2',
        'T8', 'C6', 'C4', 'C2', 'TP8', 'CP6', 'CP4', 'CP2',
        'P8', 'P6', 'P4', 'P2', 'PO8', 'PO6', 'PO4', 'CB2', 'O2'
    ]

    left_indices = [SEED_CHANNEL_NAMES.index(e) for e in left_electrodes if e in SEED_CHANNEL_NAMES]
    right_indices = [SEED_CHANNEL_NAMES.index(e) for e in right_electrodes if e in SEED_CHANNEL_NAMES]

    left_mean = electrode_scores[left_indices].mean()
    right_mean = electrode_scores[right_indices].mean()

    # Asymmetry index: (Right - Left) / (Right + Left)
    asymmetry = (right_mean - left_mean) / (right_mean + left_mean + 1e-10)

    return {
        'left_hemisphere': float(left_mean),
        'right_hemisphere': float(right_mean),
        'asymmetry_index': float(asymmetry),
        'dominant_hemisphere': 'right' if right_mean > left_mean else 'left'
    }


def select_top_electrodes(
    electrode_scores: np.ndarray,
    n_select: int = 30,
    method: str = 'top_k'
) -> Tuple[List[int], List[str]]:
    """
    Select most important electrodes for reduced-channel analysis.

    Args:
        electrode_scores: Per-electrode importance, shape (62,)
        n_select: Number of electrodes to select
        method: 'top_k' = select top K, 'threshold' = select above mean + std

    Returns:
        Tuple of (selected_indices, selected_names)
    """
    if method == 'top_k':
        selected_indices = np.argsort(electrode_scores)[::-1][:n_select]
    elif method == 'threshold':
        threshold = electrode_scores.mean() + electrode_scores.std()
        selected_indices = np.where(electrode_scores > threshold)[0]
        if len(selected_indices) < n_select:
            # Fall back to top_k if threshold selects too few
            selected_indices = np.argsort(electrode_scores)[::-1][:n_select]
    else:
        raise ValueError(f"Unknown method: {method}")

    selected_names = [SEED_CHANNEL_NAMES[i] for i in selected_indices]

    return list(selected_indices), selected_names
