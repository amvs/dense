"""
Quick test script for training pipeline.
Runs a single training job with minimal epochs to verify everything works.
"""
import argparse
import os
import sys
from configs import load_config

def parse_args():
    parser = argparse.ArgumentParser(description="Test training script with minimal configuration")
    parser.add_argument(
        "--config", type=str, default="configs/curet.yaml",
        help="Path to config file (default: configs/curet.yaml)"
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Override config values (e.g., --override classifier_epochs=1 conv_epochs=1)"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None,
        help="Wandb project name (optional)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply overrides
    from configs import apply_overrides
    config = apply_overrides(config, args.override)
    
    # Set minimal epochs for testing if not overridden
    if "classifier_epochs=" not in " ".join(args.override):
        config["classifier_epochs"] = 1
    if "conv_epochs=" not in " ".join(args.override):
        config["conv_epochs"] = 1
    
    # Create a temporary config file
    import tempfile
    import yaml
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_config_path = f.name
    
    try:
        # Build command
        cmd = [
            sys.executable, "scripts/train.py",
            "--config", temp_config_path,
        ]
        if args.wandb_project:
            cmd.extend(["--wandb_project", args.wandb_project])
        
        # Run training with output visible
        import subprocess
        result = subprocess.run(cmd, check=False)
        
        if result.returncode != 0:
            print(f"\n❌ Training failed with return code {result.returncode}")
            sys.exit(1)
        else:
            print(f"\n✅ Training completed successfully!")
            
    except Exception as e:
        print(f"\n❌ Error running training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up temp config
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)

if __name__ == "__main__":
    main()
