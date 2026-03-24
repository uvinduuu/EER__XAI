"""
Gradient-based XAI methods for EEG models

Methods:
- Saliency Maps: Simple gradient w.r.t. input
- Integrated Gradients: Path integral of gradients
- GradCAM: Gradient-weighted Class Activation Mapping
- Guided Backpropagation: Combines backprop with ReLU guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Union


class Saliency:
    """
    Saliency maps using vanilla gradients.

    Computes the gradient of the output class score with respect to input.
    Works well for identifying which input features affect the prediction.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: PyTorch model (must be in eval mode)
        """
        self.model = model
        self.model.eval()

    def attribute(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        abs_value: bool = True
    ) -> torch.Tensor:
        """
        Compute saliency map for input.

        Args:
            inputs: Input tensor of shape (batch, channels, features) or (batch, channels, bands)
            target: Target class index. If None, uses predicted class.
            abs_value: If True, return absolute value of gradients

        Returns:
            Saliency map of same shape as input
        """
        inputs = inputs.clone().requires_grad_(True)

        # Forward pass
        outputs = self.model(inputs)

        # Get target class
        if target is None:
            target = outputs.argmax(dim=1)
        elif isinstance(target, int):
            target = torch.tensor([target] * inputs.shape[0], device=inputs.device)

        # Compute gradient for target class
        one_hot = torch.zeros_like(outputs)
        one_hot.scatter_(1, target.unsqueeze(1), 1)

        outputs.backward(gradient=one_hot)

        saliency = inputs.grad.data

        if abs_value:
            saliency = torch.abs(saliency)

        return saliency

    def attribute_batch(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda',
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute saliency for entire dataset.

        Args:
            dataloader: DataLoader with test data
            device: Device to run computation on
            target_class: If specified, compute saliency for this class only

        Returns:
            Tuple of (all_saliencies, all_labels)
        """
        all_saliencies = []
        all_labels = []

        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(device).float()

            target = target_class if target_class is not None else None
            saliency = self.attribute(batch_data, target=target)

            all_saliencies.append(saliency.cpu().numpy())
            all_labels.append(batch_labels.numpy())

        return np.concatenate(all_saliencies), np.concatenate(all_labels)


class IntegratedGradients:
    """
    Integrated Gradients attribution method.

    Computes attributions by integrating gradients along a path from
    a baseline (typically zero) to the input.

    Reference: Sundararajan et al., "Axiomatic Attribution for Deep Networks"
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def attribute(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        n_steps: int = 50,
        internal_batch_size: int = 32
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients.

        Args:
            inputs: Input tensor
            target: Target class (if None, uses predicted class)
            baseline: Baseline input (if None, uses zeros)
            n_steps: Number of integration steps
            internal_batch_size: Batch size for computing gradients

        Returns:
            Attribution tensor of same shape as input
        """
        if baseline is None:
            baseline = torch.zeros_like(inputs)

        # Get target class if not specified
        if target is None:
            with torch.no_grad():
                outputs = self.model(inputs)
                target = outputs.argmax(dim=1)
        elif isinstance(target, int):
            target = torch.tensor([target] * inputs.shape[0], device=inputs.device)

        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, n_steps + 1, device=inputs.device)

        # Compute gradients for each interpolation step
        integrated_grads = torch.zeros_like(inputs)

        for i in range(0, n_steps + 1, internal_batch_size):
            batch_alphas = alphas[i:min(i + internal_batch_size, n_steps + 1)]

            # Interpolate: baseline + alpha * (input - baseline)
            batch_inputs = baseline + batch_alphas.view(-1, 1, 1, 1).expand(
                -1, *inputs.shape
            ) * (inputs - baseline)
            batch_inputs = batch_inputs.view(-1, *inputs.shape[1:])
            batch_inputs.requires_grad_(True)

            # Forward and backward
            outputs = self.model(batch_inputs)

            # Expand target for batch
            batch_target = target.repeat(len(batch_alphas))
            one_hot = torch.zeros_like(outputs)
            one_hot.scatter_(1, batch_target.unsqueeze(1), 1)

            outputs.backward(gradient=one_hot)

            # Accumulate gradients
            grads = batch_inputs.grad.view(len(batch_alphas), *inputs.shape)
            integrated_grads += grads.sum(dim=0)

        # Scale by (input - baseline) and normalize
        integrated_grads = (inputs - baseline) * integrated_grads / n_steps

        return integrated_grads

    def attribute_batch(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda',
        n_steps: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Integrated Gradients for entire dataset."""
        all_attributions = []
        all_labels = []

        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(device).float()

            attrs = self.attribute(batch_data, n_steps=n_steps)

            all_attributions.append(attrs.cpu().numpy())
            all_labels.append(batch_labels.numpy())

        return np.concatenate(all_attributions), np.concatenate(all_labels)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (GradCAM).

    For EEG models, hooks into the last convolutional layer to generate
    importance maps highlighting which spatial/channel locations contribute
    to the prediction.

    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: PyTorch model
            target_layer: The layer to compute GradCAM for (typically last conv layer)
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        # Storage for activations and gradients
        self.activations = None
        self.gradients = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def attribute(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        relu_attributions: bool = True
    ) -> torch.Tensor:
        """
        Compute GradCAM attribution.

        Args:
            inputs: Input tensor
            target: Target class (if None, uses predicted class)
            relu_attributions: If True, apply ReLU to final attributions

        Returns:
            GradCAM attribution map
        """
        # Forward pass
        outputs = self.model(inputs)

        # Get target
        if target is None:
            target = outputs.argmax(dim=1)
        elif isinstance(target, int):
            target = torch.tensor([target] * inputs.shape[0], device=inputs.device)

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(outputs)
        one_hot.scatter_(1, target.unsqueeze(1), 1)
        outputs.backward(gradient=one_hot)

        # Compute weights: global average pooling of gradients
        # Shape: (batch, channels, ...)
        weights = self.gradients.mean(dim=tuple(range(2, self.gradients.dim())), keepdim=True)

        # Weighted combination of activation maps
        gradcam = (weights * self.activations).sum(dim=1, keepdim=True)

        if relu_attributions:
            gradcam = F.relu(gradcam)

        # Normalize
        gradcam = gradcam - gradcam.min()
        if gradcam.max() > 0:
            gradcam = gradcam / gradcam.max()

        return gradcam

    def attribute_to_input_shape(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute GradCAM and resize to input shape.

        For EEG: upsamples the GradCAM to match (batch, channels, features) shape.
        """
        gradcam = self.attribute(inputs, target)

        # Resize to input spatial dimensions
        if gradcam.shape[2:] != inputs.shape[2:]:
            gradcam = F.interpolate(
                gradcam,
                size=inputs.shape[2:],
                mode='bilinear' if gradcam.dim() == 4 else 'linear',
                align_corners=False
            )

        return gradcam.squeeze(1)


class GuidedBackprop:
    """
    Guided Backpropagation.

    Modifies ReLU backward pass to only propagate positive gradients,
    producing cleaner visualizations than vanilla gradients.

    Reference: Springenberg et al., "Striving for Simplicity"
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Replace ReLU backward with guided version."""

        def guided_relu_hook(module, grad_input, grad_output):
            # Only propagate positive gradients
            return (torch.clamp(grad_input[0], min=0),)

        for module in self.model.modules():
            if isinstance(module, nn.ReLU) or isinstance(module, nn.ELU):
                hook = module.register_full_backward_hook(guided_relu_hook)
                self.hooks.append(hook)

    def remove_hooks(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def attribute(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute Guided Backpropagation.

        Args:
            inputs: Input tensor
            target: Target class

        Returns:
            Attribution map
        """
        inputs = inputs.clone().requires_grad_(True)

        outputs = self.model(inputs)

        if target is None:
            target = outputs.argmax(dim=1)
        elif isinstance(target, int):
            target = torch.tensor([target] * inputs.shape[0], device=inputs.device)

        self.model.zero_grad()
        one_hot = torch.zeros_like(outputs)
        one_hot.scatter_(1, target.unsqueeze(1), 1)
        outputs.backward(gradient=one_hot)

        return inputs.grad.data


def get_gradcam_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Get the appropriate target layer for GradCAM based on model architecture.

    Args:
        model: The PyTorch model
        model_name: Name of the model ('eegnet', 'tsception', 'dgcnn', etc.)

    Returns:
        Target layer for GradCAM
    """
    model_name = model_name.lower()

    if model_name == 'eegnet':
        # Last separable conv before pooling
        return model.sep_conv[1]

    elif model_name == 'tsception':
        # Spatial convolution layer
        if hasattr(model, 'spatialConv'):
            return model.spatialConv
        return model.conv2  # Fallback

    elif model_name == 'dgcnn':
        # Last graph conv layer
        return model.graphConvs[-1]

    elif model_name == 'acrnn':
        # After channel attention, before LSTM
        return model.conv2

    elif model_name == 'hslt':
        # Last transformer encoder
        return model.encoder.layers[-1]

    else:
        # Try to find last conv layer automatically
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                conv_layers.append(module)

        if conv_layers:
            return conv_layers[-1]
        else:
            raise ValueError(f"Could not find suitable target layer for model: {model_name}")
