"""
Utility functions for XAI module
"""

import numpy as np
import pickle
import json
import os
from pathlib import Path

# SEED-IV electrode configuration
SEED_CHANNEL_NAMES = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'OZ', 'O2', 'CB2'
]

# Brain regions for SEED-IV (9 regions)
SEED_BRAIN_REGIONS = {
    'PF': ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4'],           # Prefrontal
    'F':  ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],  # Frontal
    'LT': ['FT7', 'FC5', 'FC3', 'T7', 'C5', 'C1'],       # Left Temporal
    'RT': ['FT8', 'FC4', 'FC6', 'T8', 'C2', 'C6', 'CP6'], # Right Temporal
    'C':  ['FC1', 'C3', 'CZ', 'FCZ', 'FC2', 'C4'],       # Central
    'LP': ['TP7', 'CP5', 'CP3', 'P7', 'P5', 'P3', 'P1', 'PO3'],  # Left Parietal
    'P':  ['CP1', 'CP2', 'CPZ', 'PZ'],                    # Parietal
    'RP': ['TP8', 'CP4', 'P8', 'P6', 'P2', 'P4', 'PO4'], # Right Parietal
    'O':  ['PO7', 'PO5', 'POZ', 'PO6', 'PO8', 'CB1', 'O1', 'O2', 'OZ', 'CB2']  # Occipital
}

# Frequency bands for DE features
FREQUENCY_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 14),
    'beta': (14, 30),
    'gamma': (30, 50)
}

# Emotion labels for SEED-IV
EMOTION_LABELS = {
    0: 'neutral',
    1: 'sad',
    2: 'fear',
    3: 'happy'
}

# 2D electrode positions for topographic plotting (approximate 10-20 system)
# Normalized to [-1, 1] range for plotting
ELECTRODE_POSITIONS_2D = {
    'FP1': (-0.22, 0.95), 'FPZ': (0.0, 0.98), 'FP2': (0.22, 0.95),
    'AF3': (-0.32, 0.85), 'AF4': (0.32, 0.85),
    'F7': (-0.70, 0.70), 'F5': (-0.54, 0.72), 'F3': (-0.38, 0.73),
    'F1': (-0.19, 0.74), 'FZ': (0.0, 0.75), 'F2': (0.19, 0.74),
    'F4': (0.38, 0.73), 'F6': (0.54, 0.72), 'F8': (0.70, 0.70),
    'FT7': (-0.82, 0.50), 'FC5': (-0.62, 0.52), 'FC3': (-0.42, 0.53),
    'FC1': (-0.21, 0.54), 'FCZ': (0.0, 0.55), 'FC2': (0.21, 0.54),
    'FC4': (0.42, 0.53), 'FC6': (0.62, 0.52), 'FT8': (0.82, 0.50),
    'T7': (-0.90, 0.25), 'C5': (-0.68, 0.27), 'C3': (-0.45, 0.28),
    'C1': (-0.22, 0.29), 'CZ': (0.0, 0.30), 'C2': (0.22, 0.29),
    'C4': (0.45, 0.28), 'C6': (0.68, 0.27), 'T8': (0.90, 0.25),
    'TP7': (-0.82, 0.0), 'CP5': (-0.62, 0.02), 'CP3': (-0.42, 0.03),
    'CP1': (-0.21, 0.04), 'CPZ': (0.0, 0.05), 'CP2': (0.21, 0.04),
    'CP4': (0.42, 0.03), 'CP6': (0.62, 0.02), 'TP8': (0.82, 0.0),
    'P7': (-0.70, -0.25), 'P5': (-0.54, -0.23), 'P3': (-0.38, -0.22),
    'P1': (-0.19, -0.21), 'PZ': (0.0, -0.20), 'P2': (0.19, -0.21),
    'P4': (0.38, -0.22), 'P6': (0.54, -0.23), 'P8': (0.70, -0.25),
    'PO7': (-0.54, -0.48), 'PO5': (-0.40, -0.47), 'PO3': (-0.25, -0.46),
    'POZ': (0.0, -0.45), 'PO4': (0.25, -0.46), 'PO6': (0.40, -0.47), 'PO8': (0.54, -0.48),
    'CB1': (-0.35, -0.70), 'O1': (-0.18, -0.68), 'OZ': (0.0, -0.70),
    'O2': (0.18, -0.68), 'CB2': (0.35, -0.70)
}


def get_electrode_names():
    """Return list of 62 SEED electrode names."""
    return SEED_CHANNEL_NAMES.copy()


def get_electrode_positions():
    """Return dictionary of electrode 2D positions for plotting."""
    return ELECTRODE_POSITIONS_2D.copy()


def get_brain_regions():
    """Return dictionary mapping brain regions to electrode names."""
    return SEED_BRAIN_REGIONS.copy()


def get_electrode_to_region_map():
    """Return dictionary mapping each electrode to its brain region."""
    electrode_to_region = {}
    for region, electrodes in SEED_BRAIN_REGIONS.items():
        for electrode in electrodes:
            electrode_to_region[electrode] = region
    return electrode_to_region


def get_electrode_index(electrode_name):
    """Get the index of an electrode by name."""
    return SEED_CHANNEL_NAMES.index(electrode_name)


def get_region_indices(region_name):
    """Get indices of all electrodes in a brain region."""
    electrodes = SEED_BRAIN_REGIONS.get(region_name, [])
    return [SEED_CHANNEL_NAMES.index(e) for e in electrodes if e in SEED_CHANNEL_NAMES]


def save_xai_results(results, filepath, metadata=None):
    """
    Save XAI results to file.

    Args:
        results: dict or numpy array with XAI scores
        filepath: Path to save the results
        metadata: Optional dict with additional info (model, method, etc.)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        'results': results if isinstance(results, dict) else results.tolist(),
        'metadata': metadata or {},
        'electrode_names': SEED_CHANNEL_NAMES,
        'brain_regions': SEED_BRAIN_REGIONS
    }

    if filepath.suffix == '.pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    elif filepath.suffix == '.json':
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
    elif filepath.suffix == '.npy':
        np.save(filepath, results)
    else:
        # Default to pickle
        with open(str(filepath) + '.pkl', 'wb') as f:
            pickle.dump(save_data, f)

    print(f"XAI results saved to: {filepath}")


def load_xai_results(filepath):
    """
    Load XAI results from file.

    Args:
        filepath: Path to the saved results

    Returns:
        dict with results and metadata
    """
    filepath = Path(filepath)

    if filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.suffix == '.npy':
        return {'results': np.load(filepath)}
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def create_results_directory(base_path, model_name, method_name):
    """
    Create directory structure for saving results.

    Args:
        base_path: Base results directory
        model_name: Name of the model (e.g., 'eegnet')
        method_name: Name of XAI method (e.g., 'gradcam')

    Returns:
        dict with paths for different output types
    """
    base = Path(base_path) / model_name / method_name
    paths = {
        'visualizations': base / 'visualizations',
        'scores': base / 'scores',
        'aggregated': base / 'aggregated'
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths
