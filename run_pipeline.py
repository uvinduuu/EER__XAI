"""
Core Pipeline Functions for Training and XAI Analysis
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.io import loadmat


# =============================================================================
# DATA LOADING
# =============================================================================

def load_seed4_data(dataset_path):
    """
    Load SEED-IV dataset with DE-LDS features.

    Args:
        dataset_path: Path to SEED-IV dataset

    Returns:
        data: np.array (n_samples, 62, 5)
        labels: np.array (n_samples,)
    """
    # Session labels (SEED-IV official)
    session_labels = {
        1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
        2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
        3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
    }

    all_data = []
    all_labels = []

    # Try different path structures
    base_path = Path(dataset_path)
    if (base_path / 'eeg_feature_smooth').exists():
        feature_path = base_path / 'eeg_feature_smooth'
    elif (base_path / '1').exists():
        feature_path = base_path
    else:
        raise FileNotFoundError(f"Cannot find SEED-IV data in {dataset_path}")

    print(f"Loading data from: {feature_path}")

    for session in [1, 2, 3]:
        session_path = feature_path / str(session)
        if not session_path.exists():
            print(f"Warning: Session {session} not found")
            continue

        mat_files = sorted([f for f in os.listdir(session_path) if f.endswith('.mat')])
        print(f"Session {session}: Found {len(mat_files)} subject files")

        for mat_file in mat_files:
            file_path = session_path / mat_file
            try:
                mat_data = loadmat(str(file_path))

                for trial_idx, label in enumerate(session_labels[session]):
                    # Try different key formats
                    key = f'de_LDS{trial_idx + 1}'
                    if key not in mat_data:
                        key = f'de_movingAve{trial_idx + 1}'
                    if key not in mat_data:
                        continue

                    trial_data = mat_data[key]  # (62, n_timepoints, 5) or (n_timepoints, 62, 5)

                    # Ensure shape is (n_timepoints, 62, 5)
                    if trial_data.shape[0] == 62:
                        trial_data = trial_data.transpose(1, 0, 2)

                    # Each timepoint is a sample
                    for t in range(trial_data.shape[0]):
                        all_data.append(trial_data[t])
                        all_labels.append(label)

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    data = np.array(all_data, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    print(f"\nDataset loaded:")
    print(f"  Data shape: {data.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Classes: {np.bincount(labels)} (neutral, sad, fear, happy)")

    return data, labels


def prepare_dataloaders(data, labels, batch_size=64, train_ratio=0.7, val_ratio=0.15):
    """Create train/val/test dataloaders."""
    dataset = TensorDataset(
        torch.tensor(data, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long)
    )

    n_samples = len(dataset)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class Conv2dWithConstraint(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, max_value=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.max_value = max_value

    def forward(self, x):
        return torch.clamp(self.conv(x), max=self.max_value)


class EEGNet(nn.Module):
    """EEGNet adapted for SEED-IV (62 electrodes, 5 frequency bands)."""

    def __init__(self, num_electrodes=62, num_bands=5, num_classes=4, F1=8, D=2, dropout=0.5):
        super().__init__()
        self.F1 = F1
        self.D = D
        self.num_electrodes = num_electrodes
        self.num_bands = num_bands

        # Temporal convolution (across frequency bands)
        self.conv1 = nn.Conv2d(1, F1, (1, num_bands), padding=(0, num_bands // 2), bias=False)
        self.BN1 = nn.BatchNorm2d(F1)

        # Spatial convolution (across electrodes)
        self.depth_conv = Conv2dWithConstraint(F1, F1 * D, (num_electrodes, 1), groups=F1, bias=False)
        self.BN2 = nn.BatchNorm2d(D * F1)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 2))
        self.dropout1 = nn.Dropout(dropout)

        # Separable convolution
        F2 = D * F1
        self.sep_conv1 = nn.Conv2d(F2, F2, (1, 3), padding=(0, 1), groups=F2, bias=False)
        self.sep_conv2 = nn.Conv2d(F2, F2, 1, bias=False)
        self.BN3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 2))
        self.dropout2 = nn.Dropout(dropout)

        # Calculate flatten size dynamically
        self._fc_input_size = self._get_fc_input_size(num_electrodes, num_bands)
        self.fc = nn.Linear(self._fc_input_size, num_classes)

    def _get_fc_input_size(self, num_electrodes, num_bands):
        x = torch.zeros(1, 1, num_electrodes, num_bands)
        x = self.conv1(x)
        x = self.depth_conv(x)
        x = self.pool1(x)
        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        x = self.pool2(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        # Input: (batch, 62, 5) -> (batch, 1, 62, 5)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.BN1(x)
        x = self.depth_conv(x)
        x = self.BN2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        x = self.BN3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_model(model_name, num_electrodes=62, num_bands=5, num_classes=4):
    """Get model by name."""
    models = {
        'eegnet': lambda: EEGNet(num_electrodes, num_bands, num_classes),
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    return models[model_name]()


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, train_loader, val_loader, device, epochs=100, lr=0.001, patience=10):
    """Train the model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    return model, history


def evaluate_model(model, test_loader, device):
    """Evaluate model."""
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy, np.array(all_preds), np.array(all_labels)


# =============================================================================
# XAI ANALYSIS
# =============================================================================

def run_xai_analysis(model, test_loader, device, xai_methods, n_samples=200):
    """Run XAI analysis."""
    # Import XAI modules
    from LibEER.xai.gradient_methods import Saliency, IntegratedGradients, GradCAM
    from LibEER.xai.perturbation_methods import OcclusionAnalysis
    from LibEER.xai.aggregation import aggregate_to_electrode

    model.eval()
    results = {}

    # Prepare test data
    all_data, all_labels = [], []
    for data, labels in test_loader:
        all_data.append(data)
        all_labels.append(labels)
        if sum(d.shape[0] for d in all_data) >= n_samples:
            break

    all_data = torch.cat(all_data)[:n_samples].to(device)
    all_labels = torch.cat(all_labels)[:n_samples].numpy()

    print(f"\nRunning XAI on {len(all_data)} samples...")

    # Saliency
    if 'saliency' in xai_methods:
        print("  Computing Saliency...")
        saliency = Saliency(model)
        scores = []
        for i in tqdm(range(0, len(all_data), 32), desc="Saliency", leave=False):
            batch = all_data[i:i+32]
            s = saliency.attribute(batch)
            scores.append(s.cpu().numpy())
        scores = np.concatenate(scores)
        results['saliency'] = {
            'raw_scores': scores,
            'electrode_importance': aggregate_to_electrode(scores),
            'labels': all_labels
        }

    # Integrated Gradients
    if 'integrated_gradients' in xai_methods:
        print("  Computing Integrated Gradients...")
        ig = IntegratedGradients(model)
        scores = []
        for i in tqdm(range(0, len(all_data), 16), desc="IG", leave=False):
            batch = all_data[i:i+16]
            s = ig.attribute(batch, n_steps=30)
            scores.append(s.cpu().numpy())
        scores = np.concatenate(scores)
        results['integrated_gradients'] = {
            'raw_scores': scores,
            'electrode_importance': aggregate_to_electrode(scores),
            'labels': all_labels
        }

    # Occlusion
    if 'occlusion' in xai_methods:
        print("  Computing Occlusion...")
        occlusion = OcclusionAnalysis(model)
        scores = []
        for i in tqdm(range(0, len(all_data), 32), desc="Occlusion", leave=False):
            batch = all_data[i:i+32]
            s = occlusion.attribute(batch)
            scores.append(s.cpu().numpy())
        scores = np.concatenate(scores)
        results['occlusion'] = {
            'raw_scores': scores,
            'electrode_importance': scores.mean(axis=0),
            'labels': all_labels
        }

    # GradCAM
    if 'gradcam' in xai_methods:
        print("  Computing GradCAM...")
        target_layer = model.sep_conv2
        gradcam = GradCAM(model, target_layer)
        scores = []
        for i in tqdm(range(0, len(all_data), 32), desc="GradCAM", leave=False):
            batch = all_data[i:i+32]
            s = gradcam.attribute(batch)
            scores.append(s.cpu().numpy())
        scores = np.concatenate(scores)
        # GradCAM output may need reshaping
        if scores.ndim == 4:
            electrode_imp = scores.mean(axis=(0, 2, 3))
        else:
            electrode_imp = scores.mean(axis=0)
        results['gradcam'] = {
            'raw_scores': scores,
            'electrode_importance': electrode_imp,
            'labels': all_labels
        }

    return results


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_full_pipeline(
    dataset_path,
    model_name='eegnet',
    output_dir='./results/seed4',
    device='cuda',
    epochs=100,
    batch_size=64,
    learning_rate=0.001,
    xai_methods=None,
    skip_training=False,
    n_xai_samples=200
):
    """Run the full training and XAI pipeline."""

    if xai_methods is None:
        xai_methods = ['saliency', 'integrated_gradients', 'occlusion']

    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    print("\n" + "=" * 50)
    print("STEP 1: Loading SEED-IV Data")
    print("=" * 50)
    data, labels = load_seed4_data(dataset_path)

    # Step 2: Prepare dataloaders
    print("\n" + "=" * 50)
    print("STEP 2: Preparing DataLoaders")
    print("=" * 50)
    train_loader, val_loader, test_loader = prepare_dataloaders(data, labels, batch_size)

    # Step 3: Create model
    print("\n" + "=" * 50)
    print(f"STEP 3: Creating {model_name.upper()} Model")
    print("=" * 50)
    model = get_model(model_name)
    print(model)

    checkpoint_path = output_dir / 'model_checkpoint.pth'

    # Step 4: Train or load model
    if skip_training and checkpoint_path.exists():
        print("\n" + "=" * 50)
        print("STEP 4: Loading Existing Checkpoint")
        print("=" * 50)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        history = checkpoint.get('history', {})
    else:
        print("\n" + "=" * 50)
        print("STEP 4: Training Model")
        print("=" * 50)
        model, history = train_model(model, train_loader, val_loader, device, epochs, learning_rate)

        # Save checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'model_name': model_name,
        }, checkpoint_path)
        print(f"Checkpoint saved to: {checkpoint_path}")

    # Step 5: Evaluate
    print("\n" + "=" * 50)
    print("STEP 5: Evaluating Model")
    print("=" * 50)
    test_acc, preds, test_labels = evaluate_model(model, test_loader, device)

    # Step 6: XAI Analysis
    print("\n" + "=" * 50)
    print("STEP 6: Running XAI Analysis")
    print("=" * 50)
    xai_results = run_xai_analysis(model, test_loader, device, xai_methods, n_xai_samples)

    # Step 7: Save results
    print("\n" + "=" * 50)
    print("STEP 7: Saving Results")
    print("=" * 50)

    from LibEER.xai.utils import SEED_CHANNEL_NAMES

    for method_name, method_results in xai_results.items():
        method_dir = output_dir / method_name
        (method_dir / 'scores').mkdir(parents=True, exist_ok=True)
        (method_dir / 'visualizations').mkdir(parents=True, exist_ok=True)

        # Save scores
        np.savez(
            method_dir / 'scores' / f'{model_name}_scores.npz',
            electrode_importance=method_results['electrode_importance'],
            labels=method_results['labels']
        )

        # Save importance as JSON
        importance = method_results['electrode_importance']
        importance_dict = {
            SEED_CHANNEL_NAMES[i]: float(importance[i])
            for i in range(min(len(importance), len(SEED_CHANNEL_NAMES)))
        }
        with open(method_dir / 'scores' / f'{model_name}_importance.json', 'w') as f:
            json.dump(importance_dict, f, indent=2)

        # Generate visualization
        try:
            from LibEER.xai.visualization import plot_electrode_importance, plot_topographic_map

            plot_electrode_importance(
                importance,
                title=f"{model_name.upper()} - {method_name}",
                save_path=str(method_dir / 'visualizations' / f'{model_name}_bar.png')
            )

            plot_topographic_map(
                importance,
                title=f"{model_name.upper()} - {method_name}",
                save_path=str(method_dir / 'visualizations' / f'{model_name}_topo.png')
            )

            import matplotlib.pyplot as plt
            plt.close('all')

        except Exception as e:
            print(f"Warning: Could not generate visualization for {method_name}: {e}")

    # Print summary
    print("\n" + "=" * 50)
    print("COMPLETED!")
    print("=" * 50)
    print(f"Results saved to: {output_dir}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nTop 10 Important Electrodes (by Saliency):")
    if 'saliency' in xai_results:
        imp = xai_results['saliency']['electrode_importance']
        top_idx = np.argsort(imp)[::-1][:10]
        for i, idx in enumerate(top_idx):
            print(f"  {i+1}. {SEED_CHANNEL_NAMES[idx]}: {imp[idx]:.4f}")

    return model, xai_results
