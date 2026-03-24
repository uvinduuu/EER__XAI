"""
Run XAI Analysis on SEED-IV Dataset
====================================

Usage:
    python run_xai.py --dataset_path /kaggle/input/seed-iv --model eegnet --output_dir /kaggle/working/results

Arguments:
    --dataset_path  : Path to SEED-IV dataset folder
    --model         : Model to use (eegnet, dgcnn, acrnn, tsception)
    --output_dir    : Where to save results
    --device        : cuda or cpu (default: auto-detect)
    --epochs        : Training epochs (default: 100)
    --batch_size    : Batch size (default: 64)
    --skip_training : Skip training, load existing checkpoint
    --xai_only      : Only run XAI (requires existing checkpoint)
"""

import argparse
import os
import sys
from pathlib import Path

# Add LibEER to path
SCRIPT_DIR = Path(__file__).parent.absolute()
LIBEER_PATH = SCRIPT_DIR / 'LibEER'
sys.path.insert(0, str(LIBEER_PATH))
sys.path.insert(0, str(SCRIPT_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description='EEG Emotion Recognition with XAI Analysis')

    # Required
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to SEED-IV dataset (containing eeg_feature_smooth folder)')

    # Model selection
    parser.add_argument('--model', type=str, default='eegnet',
                        choices=['eegnet', 'dgcnn', 'acrnn', 'tsception'],
                        help='Model architecture to use')

    # Paths
    parser.add_argument('--output_dir', type=str, default='./results/seed4',
                        help='Output directory for results')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')

    # XAI options
    parser.add_argument('--xai_methods', type=str, nargs='+',
                        default=['saliency', 'integrated_gradients', 'occlusion'],
                        help='XAI methods to run')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training, load existing checkpoint')
    parser.add_argument('--xai_only', action='store_true',
                        help='Only run XAI analysis')
    parser.add_argument('--n_xai_samples', type=int, default=200,
                        help='Number of samples for XAI analysis')

    return parser.parse_args()


def main():
    args = parse_args()

    # Auto-detect device
    import torch
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("EEG Emotion Recognition with XAI Analysis")
    print("=" * 60)
    print(f"Dataset Path: {args.dataset_path}")
    print(f"Model: {args.model}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"XAI Methods: {args.xai_methods}")
    print("=" * 60)

    # Validate dataset path
    dataset_path = Path(args.dataset_path)
    feature_path = dataset_path / 'eeg_feature_smooth'

    if not feature_path.exists():
        # Try alternative structure
        if (dataset_path / '1').exists():
            feature_path = dataset_path
            print(f"Using dataset path directly: {feature_path}")
        else:
            print(f"ERROR: Cannot find eeg_feature_smooth folder in {dataset_path}")
            print("Expected structure:")
            print("  dataset_path/")
            print("    eeg_feature_smooth/")
            print("      1/  (session 1)")
            print("      2/  (session 2)")
            print("      3/  (session 3)")
            sys.exit(1)

    # Import after path setup
    from run_pipeline import run_full_pipeline

    # Run pipeline
    run_full_pipeline(
        dataset_path=str(args.dataset_path),
        model_name=args.model,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        xai_methods=args.xai_methods,
        skip_training=args.skip_training or args.xai_only,
        n_xai_samples=args.n_xai_samples
    )


if __name__ == '__main__':
    main()
