"""
Visualization functions for XAI results.

Generates publication-quality figures for:
- Topographic electrode importance maps
- Brain region importance
- Comparative analysis across methods/models
- Emotion-specific patterns
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple, Union
import os

from .utils import (
    SEED_CHANNEL_NAMES,
    SEED_BRAIN_REGIONS,
    ELECTRODE_POSITIONS_2D,
    EMOTION_LABELS,
    FREQUENCY_BANDS
)


def plot_electrode_importance(
    electrode_scores: np.ndarray,
    title: str = "Electrode Importance",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    color: str = 'steelblue',
    highlight_top_k: int = 10
) -> plt.Figure:
    """
    Bar plot of electrode importance scores.

    Args:
        electrode_scores: Importance scores for 62 electrodes
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
        color: Bar color
        highlight_top_k: Number of top electrodes to highlight

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by importance
    sorted_indices = np.argsort(electrode_scores)[::-1]
    sorted_scores = electrode_scores[sorted_indices]
    sorted_names = [SEED_CHANNEL_NAMES[i] for i in sorted_indices]

    # Create colors (highlight top k)
    colors = [color] * len(sorted_scores)
    for i in range(highlight_top_k):
        colors[i] = 'coral'

    bars = ax.bar(range(len(sorted_scores)), sorted_scores, color=colors)

    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=90, fontsize=8)
    ax.set_xlabel('Electrode', fontsize=12)
    ax.set_ylabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add legend
    legend_patches = [
        mpatches.Patch(color='coral', label=f'Top {highlight_top_k}'),
        mpatches.Patch(color=color, label='Others')
    ]
    ax.legend(handles=legend_patches, loc='upper right')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_topographic_map(
    electrode_scores: np.ndarray,
    title: str = "Electrode Importance",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    cmap: str = 'RdYlBu_r',
    show_names: bool = True,
    interpolate: bool = True
) -> plt.Figure:
    """
    Create topographic head map showing electrode importance.

    Args:
        electrode_scores: Importance scores for 62 electrodes
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        cmap: Colormap
        show_names: Whether to show electrode names
        interpolate: Whether to interpolate between electrodes

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get electrode positions
    positions = ELECTRODE_POSITIONS_2D

    # Normalize scores for coloring
    norm_scores = electrode_scores.copy()
    if norm_scores.max() != norm_scores.min():
        norm_scores = (norm_scores - norm_scores.min()) / (norm_scores.max() - norm_scores.min())

    # Create head outline
    head_circle = plt.Circle((0, 0.15), 1.05, fill=False, linewidth=2, color='black')
    ax.add_patch(head_circle)

    # Add nose
    nose = plt.Polygon([[0, 1.25], [-0.1, 1.05], [0.1, 1.05]], fill=True, color='lightgray')
    ax.add_patch(nose)

    # Add ears
    left_ear = plt.Polygon([[-1.05, 0.25], [-1.15, 0.35], [-1.15, 0.15]], fill=True, color='lightgray')
    right_ear = plt.Polygon([[1.05, 0.25], [1.15, 0.35], [1.15, 0.15]], fill=True, color='lightgray')
    ax.add_patch(left_ear)
    ax.add_patch(right_ear)

    # Plot electrodes
    x_coords = []
    y_coords = []
    colors = []

    for i, electrode in enumerate(SEED_CHANNEL_NAMES):
        if electrode in positions:
            x, y = positions[electrode]
            x_coords.append(x)
            y_coords.append(y)
            colors.append(norm_scores[i])

    # Interpolation for smooth heatmap
    if interpolate and len(x_coords) > 3:
        try:
            from scipy.interpolate import griddata
            # Create grid
            grid_x, grid_y = np.mgrid[-1.2:1.2:100j, -0.9:1.2:100j]
            grid_z = griddata(
                (x_coords, y_coords),
                colors,
                (grid_x, grid_y),
                method='cubic',
                fill_value=0
            )
            # Create head mask
            dist = np.sqrt(grid_x**2 + (grid_y - 0.15)**2)
            grid_z[dist > 1.0] = np.nan

            im = ax.imshow(
                grid_z.T,
                extent=[-1.2, 1.2, -0.9, 1.2],
                origin='lower',
                cmap=cmap,
                alpha=0.7,
                aspect='auto'
            )
            plt.colorbar(im, ax=ax, label='Normalized Importance', shrink=0.8)
        except ImportError:
            interpolate = False

    # Plot electrode positions
    scatter = ax.scatter(
        x_coords, y_coords,
        c=colors if not interpolate else 'black',
        cmap=cmap if not interpolate else None,
        s=200 if show_names else 300,
        edgecolors='black',
        linewidths=1.5,
        zorder=5
    )

    if not interpolate:
        plt.colorbar(scatter, ax=ax, label='Normalized Importance', shrink=0.8)

    # Add electrode names
    if show_names:
        for i, electrode in enumerate(SEED_CHANNEL_NAMES):
            if electrode in positions:
                x, y = positions[electrode]
                ax.annotate(
                    electrode,
                    (x, y),
                    fontsize=6,
                    ha='center',
                    va='center',
                    fontweight='bold',
                    color='white' if norm_scores[i] > 0.5 else 'black'
                )

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.1, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_region_importance(
    region_scores: Dict[str, float],
    title: str = "Brain Region Importance",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Bar plot of brain region importance.

    Args:
        region_scores: Dictionary mapping region names to scores
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    region_names = {
        'PF': 'Prefrontal',
        'F': 'Frontal',
        'LT': 'Left Temporal',
        'RT': 'Right Temporal',
        'C': 'Central',
        'LP': 'Left Parietal',
        'P': 'Parietal',
        'RP': 'Right Parietal',
        'O': 'Occipital'
    }

    regions = list(region_scores.keys())
    scores = [region_scores[r] for r in regions]
    labels = [region_names.get(r, r) for r in regions]

    # Sort by score
    sorted_idx = np.argsort(scores)[::-1]
    sorted_scores = [scores[i] for i in sorted_idx]
    sorted_labels = [labels[i] for i in sorted_idx]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_scores)))

    bars = ax.barh(range(len(sorted_scores)), sorted_scores, color=colors)
    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels, fontsize=11)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_comparative_importance(
    importance_dict: Dict[str, np.ndarray],
    title: str = "Electrode Importance Across Methods",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8),
    top_k: int = 20
) -> plt.Figure:
    """
    Compare electrode importance across multiple XAI methods or models.

    Args:
        importance_dict: Dictionary mapping method/model names to importance arrays
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        top_k: Number of top electrodes to show

    Returns:
        matplotlib Figure object
    """
    n_methods = len(importance_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize, sharey=True)

    if n_methods == 1:
        axes = [axes]

    # Get consensus top electrodes (union of top_k from each method)
    top_electrodes = set()
    for scores in importance_dict.values():
        top_electrodes.update(np.argsort(scores)[::-1][:top_k])
    top_electrodes = sorted(list(top_electrodes))

    # Normalize scores for comparison
    normalized_scores = {}
    for name, scores in importance_dict.items():
        norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        normalized_scores[name] = norm

    for i, (method_name, scores) in enumerate(normalized_scores.items()):
        ax = axes[i]

        # Get top k for this method
        sorted_idx = np.argsort(scores)[::-1][:top_k]
        sorted_scores = scores[sorted_idx]
        sorted_names = [SEED_CHANNEL_NAMES[j] for j in sorted_idx]

        colors = plt.cm.plasma(sorted_scores)
        ax.barh(range(len(sorted_scores)), sorted_scores, color=colors)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=9)
        ax.set_xlabel('Normalized Importance')
        ax.set_title(method_name, fontsize=12, fontweight='bold')
        ax.invert_yaxis()

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_emotion_specific_importance(
    emotion_scores: Dict[str, np.ndarray],
    title: str = "Emotion-Specific Electrode Importance",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    top_k: int = 15
) -> plt.Figure:
    """
    Create subplot showing electrode importance for each emotion.

    Args:
        emotion_scores: Dictionary mapping emotion names to importance arrays
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        top_k: Number of top electrodes per emotion

    Returns:
        matplotlib Figure object
    """
    emotions = list(emotion_scores.keys())
    n_emotions = len(emotions)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    emotion_colors = {
        'neutral': 'gray',
        'sad': 'blue',
        'fear': 'purple',
        'happy': 'orange'
    }

    for i, emotion in enumerate(emotions):
        if i >= 4:
            break

        ax = axes[i]
        scores = emotion_scores[emotion]

        # Normalize
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

        # Top electrodes
        sorted_idx = np.argsort(norm_scores)[::-1][:top_k]
        sorted_scores = norm_scores[sorted_idx]
        sorted_names = [SEED_CHANNEL_NAMES[j] for j in sorted_idx]

        color = emotion_colors.get(emotion, 'steelblue')
        ax.barh(range(len(sorted_scores)), sorted_scores, color=color, alpha=0.7)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=10)
        ax.set_xlabel('Normalized Importance')
        ax.set_title(f'{emotion.capitalize()}', fontsize=12, fontweight='bold')
        ax.invert_yaxis()

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_frequency_band_importance(
    band_scores: Dict[str, float],
    title: str = "Frequency Band Importance",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Bar plot of frequency band importance.

    Args:
        band_scores: Dictionary mapping band names to importance scores
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    band_labels = [
        'Delta\n(0.5-4 Hz)',
        'Theta\n(4-8 Hz)',
        'Alpha\n(8-14 Hz)',
        'Beta\n(14-30 Hz)',
        'Gamma\n(30-50 Hz)'
    ]

    scores = [band_scores.get(b, 0) for b in bands]

    bars = ax.bar(range(len(scores)), scores, color=band_colors, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(band_labels)))
    ax.set_xticklabels(band_labels, fontsize=11)
    ax.set_ylabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{score:.3f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_electrode_dropout_curve(
    dropout_results: Dict,
    title: str = "Accuracy vs. Electrodes Removed",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot accuracy curve as electrodes are progressively removed.

    Args:
        dropout_results: Dict with 'n_removed' and 'accuracy' lists
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_removed = dropout_results['n_removed']
    accuracy = dropout_results['accuracy']

    ax.plot(n_removed, accuracy, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax.fill_between(n_removed, accuracy, alpha=0.3, color='steelblue')

    ax.set_xlabel('Number of Electrodes Removed (Least Important First)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Mark baseline
    ax.axhline(y=accuracy[0], color='red', linestyle='--', alpha=0.7, label=f'Baseline: {accuracy[0]:.2%}')
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def create_all_visualizations(
    electrode_scores: np.ndarray,
    output_dir: str,
    method_name: str,
    model_name: str,
    labels: Optional[np.ndarray] = None,
    xai_scores_full: Optional[np.ndarray] = None
):
    """
    Generate all visualization types and save to output directory.

    Args:
        electrode_scores: Per-electrode importance (62,)
        output_dir: Base output directory
        method_name: Name of XAI method
        model_name: Name of model
        labels: Optional emotion labels for emotion-specific analysis
        xai_scores_full: Optional full XAI scores (samples, 62, 5) for band analysis
    """
    from .aggregation import aggregate_by_region, aggregate_by_frequency_band, aggregate_by_emotion

    os.makedirs(output_dir, exist_ok=True)

    prefix = f"{model_name}_{method_name}"

    # 1. Bar plot
    plot_electrode_importance(
        electrode_scores,
        title=f"Electrode Importance - {model_name} ({method_name})",
        save_path=os.path.join(output_dir, f"{prefix}_bar.png")
    )
    plt.close()

    # 2. Topographic map
    plot_topographic_map(
        electrode_scores,
        title=f"Electrode Importance - {model_name} ({method_name})",
        save_path=os.path.join(output_dir, f"{prefix}_topo.png")
    )
    plt.close()

    # 3. Region importance
    region_scores = aggregate_by_region(electrode_scores)
    plot_region_importance(
        region_scores,
        title=f"Brain Region Importance - {model_name} ({method_name})",
        save_path=os.path.join(output_dir, f"{prefix}_regions.png")
    )
    plt.close()

    # 4. Frequency band importance (if full scores available)
    if xai_scores_full is not None and xai_scores_full.ndim == 3:
        band_scores = aggregate_by_frequency_band(xai_scores_full)
        plot_frequency_band_importance(
            band_scores,
            title=f"Frequency Band Importance - {model_name} ({method_name})",
            save_path=os.path.join(output_dir, f"{prefix}_bands.png")
        )
        plt.close()

    # 5. Emotion-specific (if labels available)
    if labels is not None and xai_scores_full is not None:
        emotion_scores = aggregate_by_emotion(xai_scores_full, labels)
        if emotion_scores:
            plot_emotion_specific_importance(
                emotion_scores,
                title=f"Emotion-Specific Importance - {model_name} ({method_name})",
                save_path=os.path.join(output_dir, f"{prefix}_emotions.png")
            )
            plt.close()

    print(f"All visualizations saved to: {output_dir}")
