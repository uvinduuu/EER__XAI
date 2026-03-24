"""
Perturbation-based XAI methods for EEG models

Methods:
- Occlusion Analysis: Systematically mask electrodes and measure impact
- LIME: Local Interpretable Model-agnostic Explanations
- SHAP: SHapley Additive exPlanations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Union, Callable
from tqdm import tqdm


class OcclusionAnalysis:
    """
    Occlusion-based electrode importance analysis.

    Systematically masks individual electrodes (channels) and measures
    the change in model prediction. Electrodes whose removal causes
    larger prediction changes are considered more important.
    """

    def __init__(self, model: nn.Module, baseline_value: float = 0.0):
        """
        Args:
            model: PyTorch model
            baseline_value: Value to replace occluded features with (default: 0)
        """
        self.model = model
        self.model.eval()
        self.baseline_value = baseline_value

    def attribute(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        n_electrodes: int = 62
    ) -> torch.Tensor:
        """
        Compute electrode importance via occlusion.

        Args:
            inputs: Input tensor of shape (batch, channels, features)
            target: Target class for importance calculation
            n_electrodes: Number of electrodes (channels) - default 62 for SEED

        Returns:
            Importance scores for each electrode, shape (batch, n_electrodes)
        """
        batch_size = inputs.shape[0]
        device = inputs.device
        importance = torch.zeros(batch_size, n_electrodes, device=device)

        with torch.no_grad():
            # Get baseline prediction
            baseline_output = self.model(inputs)
            if target is None:
                target = baseline_output.argmax(dim=1)
            elif isinstance(target, int):
                target = torch.tensor([target] * batch_size, device=device)

            baseline_probs = torch.softmax(baseline_output, dim=1)
            baseline_confidence = baseline_probs.gather(1, target.unsqueeze(1)).squeeze()

            # Occlude each electrode and measure impact
            for electrode_idx in range(n_electrodes):
                # Create occluded input
                occluded = inputs.clone()
                occluded[:, electrode_idx, :] = self.baseline_value

                # Get prediction with occluded electrode
                occluded_output = self.model(occluded)
                occluded_probs = torch.softmax(occluded_output, dim=1)
                occluded_confidence = occluded_probs.gather(1, target.unsqueeze(1)).squeeze()

                # Importance = drop in confidence
                importance[:, electrode_idx] = baseline_confidence - occluded_confidence

        return importance

    def attribute_batch(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda',
        n_electrodes: int = 62
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute occlusion importance for entire dataset."""
        all_importance = []
        all_labels = []

        for batch_data, batch_labels in tqdm(dataloader, desc="Occlusion Analysis"):
            batch_data = batch_data.to(device).float()

            importance = self.attribute(batch_data, n_electrodes=n_electrodes)

            all_importance.append(importance.cpu().numpy())
            all_labels.append(batch_labels.numpy())

        return np.concatenate(all_importance), np.concatenate(all_labels)

    def attribute_per_band(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        n_electrodes: int = 62,
        n_bands: int = 5
    ) -> torch.Tensor:
        """
        Compute importance for each electrode-band combination.

        Args:
            inputs: Input tensor of shape (batch, channels, bands)
            target: Target class
            n_electrodes: Number of electrodes
            n_bands: Number of frequency bands

        Returns:
            Importance scores, shape (batch, n_electrodes, n_bands)
        """
        batch_size = inputs.shape[0]
        device = inputs.device
        importance = torch.zeros(batch_size, n_electrodes, n_bands, device=device)

        with torch.no_grad():
            # Baseline
            baseline_output = self.model(inputs)
            if target is None:
                target = baseline_output.argmax(dim=1)
            elif isinstance(target, int):
                target = torch.tensor([target] * batch_size, device=device)

            baseline_probs = torch.softmax(baseline_output, dim=1)
            baseline_conf = baseline_probs.gather(1, target.unsqueeze(1)).squeeze()

            for e in range(n_electrodes):
                for b in range(n_bands):
                    occluded = inputs.clone()
                    occluded[:, e, b] = self.baseline_value

                    occluded_output = self.model(occluded)
                    occluded_probs = torch.softmax(occluded_output, dim=1)
                    occluded_conf = occluded_probs.gather(1, target.unsqueeze(1)).squeeze()

                    importance[:, e, b] = baseline_conf - occluded_conf

        return importance


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) for EEG.

    Creates local linear approximations of the model behavior by
    perturbing the input and fitting a simple interpretable model.

    For EEG: treats each electrode as a "superpixel" and measures
    contribution by selectively masking channels.
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 1000,
        kernel_width: float = 0.25
    ):
        """
        Args:
            model: PyTorch model
            n_samples: Number of perturbed samples to generate
            kernel_width: Width of exponential kernel for weighting samples
        """
        self.model = model
        self.model.eval()
        self.n_samples = n_samples
        self.kernel_width = kernel_width

    def _kernel(self, distances: np.ndarray) -> np.ndarray:
        """Exponential kernel for weighting samples."""
        return np.exp(-(distances ** 2) / (self.kernel_width ** 2))

    def attribute(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        n_electrodes: int = 62
    ) -> np.ndarray:
        """
        Compute LIME explanation for input sample.

        Args:
            inputs: Single input tensor of shape (1, channels, features)
            target: Target class for explanation
            n_electrodes: Number of electrodes

        Returns:
            LIME coefficients for each electrode
        """
        assert inputs.shape[0] == 1, "LIME works on single samples"

        device = inputs.device
        inputs_np = inputs.cpu().numpy().squeeze()

        # Get target class if not specified
        with torch.no_grad():
            output = self.model(inputs)
            if target is None:
                target = output.argmax().item()

        # Generate perturbed samples
        # Binary mask for each electrode (1 = keep, 0 = occlude)
        masks = np.random.randint(0, 2, size=(self.n_samples, n_electrodes))
        masks[0] = 1  # First sample is the original

        # Create perturbed inputs
        perturbed_inputs = []
        for mask in masks:
            perturbed = inputs_np.copy()
            for i, keep in enumerate(mask):
                if not keep:
                    perturbed[i, :] = 0  # Occlude electrode
            perturbed_inputs.append(perturbed)

        perturbed_inputs = np.array(perturbed_inputs)
        perturbed_tensor = torch.tensor(perturbed_inputs, device=device, dtype=torch.float32)

        # Get predictions for perturbed inputs
        with torch.no_grad():
            outputs = self.model(perturbed_tensor)
            probs = torch.softmax(outputs, dim=1)[:, target].cpu().numpy()

        # Compute distances (Hamming distance from original)
        distances = np.sum(masks == 0, axis=1) / n_electrodes

        # Compute weights using kernel
        weights = self._kernel(distances)

        # Fit weighted linear regression
        # y = probs, X = masks (binary), weighted by kernel
        from sklearn.linear_model import Ridge

        model_lr = Ridge(alpha=1.0)
        model_lr.fit(masks, probs, sample_weight=weights)

        return model_lr.coef_

    def attribute_batch(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda',
        n_electrodes: int = 62
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute LIME for multiple samples."""
        all_importance = []
        all_labels = []

        for batch_data, batch_labels in tqdm(dataloader, desc="LIME Analysis"):
            batch_data = batch_data.to(device).float()

            for i in range(batch_data.shape[0]):
                single_input = batch_data[i:i+1]
                importance = self.attribute(single_input, n_electrodes=n_electrodes)
                all_importance.append(importance)
                all_labels.append(batch_labels[i].item())

        return np.array(all_importance), np.array(all_labels)


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) for EEG models.

    Computes exact Shapley values by considering all possible
    coalitions of electrodes. For efficiency, uses sampling-based
    approximation.

    For 62 electrodes, uses KernelSHAP approach.
    """

    def __init__(
        self,
        model: nn.Module,
        background_data: Optional[torch.Tensor] = None,
        n_samples: int = 100
    ):
        """
        Args:
            model: PyTorch model
            background_data: Background dataset for computing expected values
            n_samples: Number of samples for SHAP approximation
        """
        self.model = model
        self.model.eval()
        self.background_data = background_data
        self.n_samples = n_samples

    def _predict(self, inputs: torch.Tensor) -> np.ndarray:
        """Get model predictions as probabilities."""
        with torch.no_grad():
            outputs = self.model(inputs)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    def attribute(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        n_electrodes: int = 62
    ) -> np.ndarray:
        """
        Compute SHAP values for input sample.

        Uses sampling-based Shapley value approximation.

        Args:
            inputs: Single input tensor of shape (1, channels, features)
            target: Target class
            n_electrodes: Number of electrodes

        Returns:
            SHAP values for each electrode
        """
        assert inputs.shape[0] == 1, "SHAP works on single samples"

        device = inputs.device
        input_np = inputs.cpu().numpy().squeeze()

        # Get baseline (expected value with no features)
        if self.background_data is not None:
            baseline = self.background_data.mean(dim=0).cpu().numpy()
        else:
            baseline = np.zeros_like(input_np)

        # Get target class
        with torch.no_grad():
            output = self.model(inputs)
            if target is None:
                target = output.argmax().item()
            base_pred = torch.softmax(output, dim=1)[0, target].item()

        # Monte Carlo sampling of Shapley values
        shap_values = np.zeros(n_electrodes)
        marginal_contributions = [[] for _ in range(n_electrodes)]

        for _ in range(self.n_samples):
            # Random permutation of electrodes
            perm = np.random.permutation(n_electrodes)

            # Start with baseline
            current = baseline.copy()
            current_pred = None

            for i, electrode_idx in enumerate(perm):
                # Add this electrode
                prev = current.copy()
                current[electrode_idx] = input_np[electrode_idx]

                # Compute predictions
                prev_tensor = torch.tensor(prev, device=device, dtype=torch.float32).unsqueeze(0)
                curr_tensor = torch.tensor(current, device=device, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    if current_pred is None:
                        prev_pred = self._predict(prev_tensor)[0, target]
                    else:
                        prev_pred = current_pred
                    current_pred = self._predict(curr_tensor)[0, target]

                # Marginal contribution
                contribution = current_pred - prev_pred
                marginal_contributions[electrode_idx].append(contribution)

        # Average marginal contributions = Shapley values
        for i in range(n_electrodes):
            if marginal_contributions[i]:
                shap_values[i] = np.mean(marginal_contributions[i])

        return shap_values

    def attribute_batch(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda',
        n_electrodes: int = 62
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute SHAP values for multiple samples."""
        all_shap = []
        all_labels = []

        for batch_data, batch_labels in tqdm(dataloader, desc="SHAP Analysis"):
            batch_data = batch_data.to(device).float()

            for i in range(batch_data.shape[0]):
                single_input = batch_data[i:i+1]
                shap_vals = self.attribute(single_input, n_electrodes=n_electrodes)
                all_shap.append(shap_vals)
                all_labels.append(batch_labels[i].item())

        return np.array(all_shap), np.array(all_labels)


def compute_electrode_dropout_curve(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    importance_scores: np.ndarray,
    device: str = 'cuda',
    n_electrodes: int = 62
) -> dict:
    """
    Compute accuracy curve as electrodes are progressively removed.

    Useful for validating that XAI-identified important electrodes
    actually contribute to model performance.

    Args:
        model: Trained model
        test_loader: Test data loader
        importance_scores: Electrode importance scores (shape: (n_electrodes,))
        device: Device to run on
        n_electrodes: Number of electrodes

    Returns:
        dict with 'n_removed', 'accuracy', 'electrode_names' removed at each step
    """
    from .utils import get_electrode_names

    model.eval()
    electrode_names = get_electrode_names()

    # Sort electrodes by importance (least to most important)
    sorted_indices = np.argsort(importance_scores)
    sorted_names = [electrode_names[i] for i in sorted_indices]

    results = {
        'n_removed': [0],
        'accuracy': [],
        'removed_electrodes': [[]]
    }

    # Compute baseline accuracy (no electrodes removed)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device).float()
            labels = labels.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    results['accuracy'].append(correct / total)

    # Progressively remove electrodes (least important first)
    for n_remove in range(5, n_electrodes, 5):
        electrodes_to_remove = sorted_indices[:n_remove]

        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device).float()
                # Zero out removed electrodes
                data[:, electrodes_to_remove, :] = 0
                labels = labels.to(device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        results['n_removed'].append(n_remove)
        results['accuracy'].append(correct / total)
        results['removed_electrodes'].append(sorted_names[:n_remove])

    return results
