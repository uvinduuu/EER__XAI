# XAI for EEG Emotion Recognition - Kaggle Workflow Guide

## What You Need from Model Training for XAI

### Essential Artifacts to Save

| Artifact | Why It's Needed | How to Save |
|----------|-----------------|-------------|
| **Model Weights** | To reload the trained model | `torch.save(model.state_dict(), 'model.pth')` |
| **Model Config** | To recreate the architecture | Save hyperparameters (F1, D, etc.) |
| **Test Data Sample** | Input for generating explanations | Save as `.npz` file |

### Which Layer for Each XAI Method?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         XAI METHOD → LAYER MAPPING                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  EEGNet Architecture:                                                        │
│  Input(62,5) → Conv1 → DepthConv → SepConv1 → SepConv2 → Pool → FC → Output │
│       │                                   │      │              │            │
│       │                                   │      │              │            │
│       ▼                                   ▼      ▼              ▼            │
│  ┌─────────────┐                    ┌──────────┐          ┌──────────┐      │
│  │  SALIENCY   │                    │ GRADCAM  │          │   FC     │      │
│  │  Integrated │                    │ (hooks   │          │ weights  │      │
│  │  Gradients  │                    │  here)   │          │          │      │
│  │  SHAP/LIME  │                    │          │          │          │      │
│  └─────────────┘                    └──────────┘          └──────────┘      │
│   ↑ Gradients                        ↑ Last conv           ↑ For linear    │
│   w.r.t INPUT                        before FC              importance      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

CRITICAL UNDERSTANDING:
- Saliency/IntegratedGradients: Compute gradients w.r.t. INPUT → gives (62, 5) importance
- GradCAM: Hooks into LAST CONV LAYER → upsamples to input shape
- SHAP/LIME: Treat model as BLACK BOX → perturb input, observe output
- Occlusion: Zero out each electrode → measure accuracy drop
```

## Kaggle Workflow

### Notebook 1: Training (GPU - P100/T4 recommended)
```python
# 1. Train model
model, history = train_model(model, train_loader, val_loader, config)

# 2. SAVE EVERYTHING YOU NEED
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {...},  # F1, D, dropout, n_classes
}, 'checkpoint.pth')

# 3. Save test samples for XAI
np.savez('test_samples.npz', data=test_data, labels=test_labels)
```

### Notebook 2: XAI Analysis (Can run on CPU, but GPU faster)
```python
# 1. Load model
checkpoint = torch.load('checkpoint.pth')
model = EEGNet(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. Load test data
samples = np.load('test_samples.npz')
test_data = torch.tensor(samples['data'])

# 3. Run XAI methods
saliency = Saliency(model)
scores = saliency.attribute(test_data)  # Shape: (n_samples, 62, 5)

# 4. Aggregate to electrode level
electrode_importance = scores.abs().mean(dim=(0, 2))  # Shape: (62,)
```

## Output Structure

```
results/seed4/
├── eegnet/
│   ├── model_checkpoint.pth
│   ├── training_history.json
│   ├── test_samples.npz
│   ├── saliency/
│   │   ├── scores/
│   │   │   ├── eegnet_scores.npz
│   │   │   └── eegnet_importance.json
│   │   └── visualizations/
│   │       ├── eegnet_saliency_bar.png
│   │       ├── eegnet_saliency_topo.png
│   │       └── eegnet_saliency_regions.png
│   ├── integrated_gradients/
│   ├── gradcam/
│   ├── occlusion/
│   ├── lime/
│   └── shap/
├── dgcnn/
├── acrnn/
├── comparative_analysis/
│   └── cross_method_comparison.png
└── channel_selection/
    └── reduced_channel_results.json
```

## Quick Start Commands

### On Kaggle:
```bash
# Clone the repo (or upload as dataset)
# Install requirements
pip install captum scikit-learn scipy matplotlib

# Run training notebook
python 01_eegnet_train_xai.py
```

### Estimated Runtimes (Kaggle P100):
| Step | Time |
|------|------|
| Training (100 epochs) | 10-15 min |
| Saliency (200 samples) | 1-2 min |
| Integrated Gradients | 5-10 min |
| GradCAM | 2-3 min |
| Occlusion | 5-10 min |
| LIME (50 samples) | 15-20 min |
| SHAP (30 samples) | 20-30 min |
| **Total** | **~1 hour** |

## XAI Method Recommendations

| Method | Speed | Quality | Best For |
|--------|-------|---------|----------|
| **Saliency** | Very Fast | Good | Quick analysis, baseline |
| **Integrated Gradients** | Medium | Very Good | Publication-quality results |
| **GradCAM** | Fast | Good | Spatial visualization |
| **Occlusion** | Medium | Very Good | Intuitive interpretation |
| **LIME** | Slow | Good | Model-agnostic validation |
| **SHAP** | Very Slow | Excellent | Theoretical soundness |

**For Paper:** Use Integrated Gradients + Occlusion as primary, validate with SHAP on subset.
