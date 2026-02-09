"""
Verify training checkpoints and epoch information.
Usage: python verify_training.py --model_dir <path> [--num_models N]
"""
import argparse
import os
import torch

def verify_training(model_dir, num_models=5):
    print("=" * 80)
    print("Training Verification")
    print("=" * 80)
    print(f"Directory: {model_dir}\n")

    if not os.path.exists(model_dir):
        print(f"ERROR: Directory not found: {model_dir}")
        return

    for i in range(num_models):
        model_path = os.path.join(model_dir, f'model_{i}', 'model.pt')

        if not os.path.exists(model_path):
            print(f"Model {i}: NOT FOUND")
            continue

        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"\nModel {i}:")

            if 'epoch' in checkpoint:
                print(f"  Trained epochs: {checkpoint['epoch']}")

            if 'args' in checkpoint:
                args = checkpoint['args']
                if hasattr(args, 'epochs'):
                    print(f"  Configured epochs: {args.epochs}")
                if hasattr(args, 'dataset_type'):
                    print(f"  Dataset type: {args.dataset_type}")

            if 'state_dict' in checkpoint:
                print(f"  Model weights: Present")

            print(f"  Available keys: {list(checkpoint.keys())}")

        except Exception as e:
            print(f"\nModel {i}: ERROR - {e}")

    print("\n" + "=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify training progress')
    parser.add_argument('--model_dir', required=True,
                        help='Directory containing model checkpoints')
    parser.add_argument('--num_models', type=int, default=5,
                        help='Number of models to check')

    args = parser.parse_args()
    verify_training(args.model_dir, args.num_models)
