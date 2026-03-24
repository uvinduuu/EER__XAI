"""
EEGNet Training + XAI Analysis Pipeline for SEED-IV
====================================================

This script demonstrates the complete workflow:
1. Train EEGNet on SEED-IV dataset
2. Save model weights and artifacts
3. Run all XAI methods (Saliency, IntegratedGradients, GradCAM, SHAP, LIME, Occlusion)
4. Aggregate results to electrode level
5. Generate visualizations

For Kaggle:
- Upload SEED-IV dataset to Kaggle Datasets
- Run this notebook with GPU enabled (P100 or T4)
- Results will be saved to /kaggle/working/
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# Add LibEER to path
sys.path.append('/kaggle/input/libeer')  # Adjust path based on your setup
# Or for local: sys.path.append('d:/Work/Dr. Asanka/MerCon Paper/EER__XAI/LibEER')

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Dataset
    'dataset_path': '/kaggle/input/seed-iv',  # Adjust for your Kaggle dataset
    'feature_type': 'de_lds',  # Differential Entropy with LDS smoothing
    'n_electrodes': 62,
    'n_bands': 5,
    'n_classes': 4,

    # Model
    'model_name': 'eegnet',
    'F1': 8,
    'D': 2,
    'dropout': 0.5,

    # Training
    'batch_size': 64,
    'learning_rate': 0.001,
    'epochs': 100,
    'early_stopping_patience': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # XAI
    'xai_methods': ['saliency', 'integrated_gradients', 'gradcam', 'occlusion', 'lime', 'shap'],
    'n_samples_lime': 500,
    'n_samples_shap': 50,
    'n_steps_ig': 50,

    # Output
    'output_dir': '/kaggle/working/results/seed4/eegnet',
    'save_model': True,
    'save_xai_scores': True,
    'generate_visualizations': True,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_seed4_data(dataset_path, feature_type='de_lds'):
    """
    Load SEED-IV dataset with DE features.

    Expected structure:
    dataset_path/
        eeg_feature_smooth/
            1/  # Session 1
                1_20131027.mat  # Subject files
                ...
            2/  # Session 2
            3/  # Session 3

    Returns:
        data: np.array of shape (n_samples, 62, 5)
        labels: np.array of shape (n_samples,)
    """
    from scipy.io import loadmat

    # Session labels (from LibEER)
    session_labels = {
        1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
        2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
        3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    }

    all_data = []
    all_labels = []

    feature_path = os.path.join(dataset_path, 'eeg_feature_smooth')

    for session in [1, 2, 3]:
        session_path = os.path.join(feature_path, str(session))
        if not os.path.exists(session_path):
            print(f"Warning: Session {session} not found at {session_path}")
            continue

        for mat_file in sorted(os.listdir(session_path)):
            if not mat_file.endswith('.mat'):
                continue

            file_path = os.path.join(session_path, mat_file)
            try:
                mat_data = loadmat(file_path)

                # Extract DE features from each trial
                for trial_idx, label in enumerate(session_labels[session]):
                    # Key format: de_LDS1, de_LDS2, ..., de_LDS24
                    key = f'de_LDS{trial_idx + 1}'
                    if key not in mat_data:
                        continue

                    trial_data = mat_data[key]  # Shape: (62, n_timepoints, 5)

                    # Transpose to (n_timepoints, 62, 5) if needed
                    if trial_data.shape[0] == 62:
                        trial_data = trial_data.transpose(1, 0, 2)

                    # Each timepoint is a sample
                    for t in range(trial_data.shape[0]):
                        all_data.append(trial_data[t])  # (62, 5)
                        all_labels.append(label)

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    data = np.array(all_data, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    print(f"Loaded data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Class distribution: {np.bincount(labels)}")

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


# =============================================================================
# MODEL DEFINITION
# =============================================================================

class Conv2dWithConstraint(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, max_value=1.0, bias=False, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.max_value = max_value

    def forward(self, x):
        return torch.clamp(self.conv(x), max=self.max_value)


class EEGNet(nn.Module):
    """EEGNet for SEED-IV (62 channels, 5 frequency bands as "time" dimension)."""

    def __init__(self, num_electrodes=62, datapoints=5, num_classes=4, F1=8, D=2, dropout=0.5):
        super().__init__()
        self.F1 = F1
        self.D = D

        # Temporal conv (across frequency bands)
        self.conv1 = nn.Conv2d(1, F1, (1, datapoints), padding='same', bias=False)
        self.BN1 = nn.BatchNorm2d(F1)

        # Spatial conv (across electrodes)
        self.depth_conv = Conv2dWithConstraint(
            F1, F1 * D, (num_electrodes, 1), bias=False, groups=F1
        )
        self.BN2 = nn.BatchNorm2d(D * F1)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 2))
        self.dropout1 = nn.Dropout(dropout)

        # Separable conv
        F2 = D * F1
        self.sep_conv1 = nn.Conv2d(F2, F2, (1, 3), padding='same', groups=F2, bias=False)
        self.sep_conv2 = nn.Conv2d(F2, F2, 1, bias=False)  # This is our GradCAM target
        self.BN3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 2))
        self.dropout2 = nn.Dropout(dropout)

        # Calculate FC input size
        # After pools: 5 -> 2 -> 1 (approximately)
        self.fc = nn.Linear(F2, num_classes)

    def forward(self, x):
        # x: (batch, 62, 5) -> reshape to (batch, 1, 62, 5)
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


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, train_loader, val_loader, config):
    """Train the model and return training history."""
    device = config['device']
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
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

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

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

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_state)
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    return model, history


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

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
# SAVE ARTIFACTS (WHAT YOU NEED FOR XAI)
# =============================================================================

def save_training_artifacts(model, history, config, test_loader, output_dir):
    """
    Save all artifacts needed for XAI analysis.

    THIS IS CRITICAL - save these during training on Kaggle:
    1. Model weights (state_dict)
    2. Model config (architecture parameters)
    3. Training history
    4. Sample test data for XAI
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save model weights
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_electrodes': config['n_electrodes'],
            'datapoints': config['n_bands'],
            'num_classes': config['n_classes'],
            'F1': config['F1'],
            'D': config['D'],
            'dropout': config['dropout'],
        },
        'config': config,
    }, output_dir / 'model_checkpoint.pth')

    print(f"Model saved to: {output_dir / 'model_checkpoint.pth'}")

    # 2. Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # 3. Save sample test data for XAI (first 500 samples)
    test_data = []
    test_labels = []
    for data, labels in test_loader:
        test_data.append(data.numpy())
        test_labels.append(labels.numpy())
        if len(test_data) * data.shape[0] >= 500:
            break

    test_data = np.concatenate(test_data)[:500]
    test_labels = np.concatenate(test_labels)[:500]

    np.savez(
        output_dir / 'test_samples.npz',
        data=test_data,
        labels=test_labels
    )
    print(f"Test samples saved to: {output_dir / 'test_samples.npz'}")


# =============================================================================
# XAI ANALYSIS
# =============================================================================

def run_xai_analysis(model, test_loader, config):
    """
    Run all XAI methods and return results.

    Returns:
        dict: XAI results for each method
    """
    device = config['device']
    model = model.to(device)
    model.eval()

    # Import XAI module
    from xai.gradient_methods import Saliency, IntegratedGradients, GradCAM, get_gradcam_target_layer
    from xai.perturbation_methods import OcclusionAnalysis, LIMEExplainer, SHAPExplainer
    from xai.aggregation import aggregate_to_electrode

    results = {}

    # Prepare test data
    all_data = []
    all_labels = []
    for data, labels in test_loader:
        all_data.append(data)
        all_labels.append(labels)
    all_data = torch.cat(all_data).to(device)
    all_labels = torch.cat(all_labels).numpy()

    # Limit samples for expensive methods
    n_samples = min(200, len(all_data))
    sample_data = all_data[:n_samples]
    sample_labels = all_labels[:n_samples]

    print(f"Running XAI on {n_samples} samples...")

    # 1. Saliency
    if 'saliency' in config['xai_methods']:
        print("Computing Saliency...")
        saliency = Saliency(model)
        saliency_scores = []
        for i in tqdm(range(0, n_samples, 32)):
            batch = sample_data[i:i+32]
            scores = saliency.attribute(batch)
            saliency_scores.append(scores.cpu().numpy())
        saliency_scores = np.concatenate(saliency_scores)
        results['saliency'] = {
            'raw_scores': saliency_scores,
            'electrode_importance': aggregate_to_electrode(saliency_scores),
            'labels': sample_labels
        }

    # 2. Integrated Gradients
    if 'integrated_gradients' in config['xai_methods']:
        print("Computing Integrated Gradients...")
        ig = IntegratedGradients(model)
        ig_scores = []
        for i in tqdm(range(0, n_samples, 16)):
            batch = sample_data[i:i+16]
            scores = ig.attribute(batch, n_steps=config['n_steps_ig'])
            ig_scores.append(scores.cpu().numpy())
        ig_scores = np.concatenate(ig_scores)
        results['integrated_gradients'] = {
            'raw_scores': ig_scores,
            'electrode_importance': aggregate_to_electrode(ig_scores),
            'labels': sample_labels
        }

    # 3. GradCAM
    if 'gradcam' in config['xai_methods']:
        print("Computing GradCAM...")
        target_layer = model.sep_conv2  # Last conv before FC
        gradcam = GradCAM(model, target_layer)
        gradcam_scores = []
        for i in tqdm(range(0, n_samples, 32)):
            batch = sample_data[i:i+32]
            scores = gradcam.attribute(batch)
            gradcam_scores.append(scores.cpu().numpy())
        gradcam_scores = np.concatenate(gradcam_scores)
        results['gradcam'] = {
            'raw_scores': gradcam_scores,
            'electrode_importance': gradcam_scores.mean(axis=(0, 2, 3)) if gradcam_scores.ndim == 4 else gradcam_scores.mean(axis=0),
            'labels': sample_labels
        }

    # 4. Occlusion
    if 'occlusion' in config['xai_methods']:
        print("Computing Occlusion...")
        occlusion = OcclusionAnalysis(model)
        occ_scores = []
        for i in tqdm(range(0, n_samples, 32)):
            batch = sample_data[i:i+32]
            scores = occlusion.attribute(batch, n_electrodes=config['n_electrodes'])
            occ_scores.append(scores.cpu().numpy())
        occ_scores = np.concatenate(occ_scores)
        results['occlusion'] = {
            'raw_scores': occ_scores,
            'electrode_importance': occ_scores.mean(axis=0),
            'labels': sample_labels
        }

    # 5. LIME (slower - use fewer samples)
    if 'lime' in config['xai_methods']:
        print("Computing LIME...")
        n_lime = min(50, n_samples)
        lime = LIMEExplainer(model, n_samples=config['n_samples_lime'])
        lime_scores = []
        for i in tqdm(range(n_lime)):
            sample = sample_data[i:i+1]
            scores = lime.attribute(sample, n_electrodes=config['n_electrodes'])
            lime_scores.append(scores)
        lime_scores = np.array(lime_scores)
        results['lime'] = {
            'raw_scores': lime_scores,
            'electrode_importance': lime_scores.mean(axis=0),
            'labels': sample_labels[:n_lime]
        }

    # 6. SHAP (slowest - use even fewer samples)
    if 'shap' in config['xai_methods']:
        print("Computing SHAP...")
        n_shap = min(30, n_samples)
        shap = SHAPExplainer(model, n_samples=config['n_samples_shap'])
        shap_scores = []
        for i in tqdm(range(n_shap)):
            sample = sample_data[i:i+1]
            scores = shap.attribute(sample, n_electrodes=config['n_electrodes'])
            shap_scores.append(scores)
        shap_scores = np.array(shap_scores)
        results['shap'] = {
            'raw_scores': shap_scores,
            'electrode_importance': shap_scores.mean(axis=0),
            'labels': sample_labels[:n_shap]
        }

    return results


def save_xai_results(results, output_dir, model_name):
    """Save XAI results to files."""
    output_dir = Path(output_dir)

    for method_name, method_results in results.items():
        method_dir = output_dir / method_name
        method_dir.mkdir(parents=True, exist_ok=True)

        # Save scores
        np.savez(
            method_dir / 'scores' / f'{model_name}_scores.npz',
            raw_scores=method_results['raw_scores'],
            electrode_importance=method_results['electrode_importance'],
            labels=method_results['labels']
        )

        # Save electrode importance as JSON for easy viewing
        from xai.utils import SEED_CHANNEL_NAMES
        importance_dict = {
            name: float(method_results['electrode_importance'][i])
            for i, name in enumerate(SEED_CHANNEL_NAMES[:len(method_results['electrode_importance'])])
        }
        with open(method_dir / 'scores' / f'{model_name}_importance.json', 'w') as f:
            json.dump(importance_dict, f, indent=2)

    print(f"XAI results saved to: {output_dir}")


def generate_visualizations(results, output_dir, model_name):
    """Generate all visualizations."""
    from xai.visualization import create_all_visualizations

    output_dir = Path(output_dir)

    for method_name, method_results in results.items():
        vis_dir = output_dir / method_name / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)

        create_all_visualizations(
            electrode_scores=method_results['electrode_importance'],
            output_dir=str(vis_dir),
            method_name=method_name,
            model_name=model_name,
            labels=method_results.get('labels'),
            xai_scores_full=method_results.get('raw_scores')
        )

    print(f"Visualizations saved to: {output_dir}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 60)
    print("EEGNet Training + XAI Analysis Pipeline")
    print("=" * 60)

    # Create output directories
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    print("\n[1/5] Loading SEED-IV data...")
    data, labels = load_seed4_data(CONFIG['dataset_path'], CONFIG['feature_type'])

    # Step 2: Prepare dataloaders
    print("\n[2/5] Preparing dataloaders...")
    train_loader, val_loader, test_loader = prepare_dataloaders(
        data, labels, CONFIG['batch_size']
    )

    # Step 3: Create and train model
    print("\n[3/5] Training EEGNet...")
    model = EEGNet(
        num_electrodes=CONFIG['n_electrodes'],
        datapoints=CONFIG['n_bands'],
        num_classes=CONFIG['n_classes'],
        F1=CONFIG['F1'],
        D=CONFIG['D'],
        dropout=CONFIG['dropout']
    )

    model, history = train_model(model, train_loader, val_loader, CONFIG)

    # Evaluate
    test_acc, preds, labels = evaluate_model(model, test_loader, CONFIG['device'])

    # Step 4: Save artifacts
    print("\n[4/5] Saving training artifacts...")
    save_training_artifacts(model, history, CONFIG, test_loader, output_dir)

    # Step 5: Run XAI
    print("\n[5/5] Running XAI analysis...")
    xai_results = run_xai_analysis(model, test_loader, CONFIG)

    # Save XAI results
    if CONFIG['save_xai_scores']:
        save_xai_results(xai_results, output_dir, CONFIG['model_name'])

    # Generate visualizations
    if CONFIG['generate_visualizations']:
        generate_visualizations(xai_results, output_dir, CONFIG['model_name'])

    print("\n" + "=" * 60)
    print("COMPLETED!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    # Return for further analysis
    return model, xai_results


if __name__ == '__main__':
    model, results = main()
