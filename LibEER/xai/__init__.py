"""
XAI Module for LibEER - Explainable AI for EEG Emotion Recognition

This module provides various XAI methods for analyzing electrode importance
in EEG-based emotion recognition models.

Methods included:
- Gradient-based: Saliency, Integrated Gradients, GradCAM
- Perturbation-based: SHAP, LIME, Occlusion
- Attention extraction: For ACRNN, HSLT models

Author: XAI Integration for SEED-IV Analysis
"""

from .gradient_methods import (
    Saliency,
    IntegratedGradients,
    GradCAM,
    GuidedBackprop
)

from .perturbation_methods import (
    OcclusionAnalysis,
    LIMEExplainer,
    SHAPExplainer
)

from .aggregation import (
    aggregate_to_electrode,
    aggregate_by_region,
    aggregate_by_frequency_band
)

from .visualization import (
    plot_electrode_importance,
    plot_topographic_map,
    plot_comparative_importance,
    plot_emotion_specific_importance
)

from .utils import (
    get_electrode_names,
    get_electrode_positions,
    get_brain_regions,
    save_xai_results,
    load_xai_results
)

__all__ = [
    # Gradient methods
    'Saliency',
    'IntegratedGradients',
    'GradCAM',
    'GuidedBackprop',
    # Perturbation methods
    'OcclusionAnalysis',
    'LIMEExplainer',
    'SHAPExplainer',
    # Aggregation
    'aggregate_to_electrode',
    'aggregate_by_region',
    'aggregate_by_frequency_band',
    # Visualization
    'plot_electrode_importance',
    'plot_topographic_map',
    'plot_comparative_importance',
    'plot_emotion_specific_importance',
    # Utils
    'get_electrode_names',
    'get_electrode_positions',
    'get_brain_regions',
    'save_xai_results',
    'load_xai_results'
]
