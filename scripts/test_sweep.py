"""
Quick test script for sweep pipeline.
Tests the sweep setup with minimal configurations.
"""
import argparse
import sys
import subprocess
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Test sweep script")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to sweep config file"
    )
    parser.add_argument(
        "--test_runs", type=int, default=1,
        help="Number of runs to test (default: 1)"
    )
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="GPU IDs to use (e.g., '0' or '0,1')"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Build command
    cmd = [
        sys.executable, "scripts/sweep.py",
        "--config", args.config,
        "--test",
        "--test_runs", str(args.test_runs),
    ]
    
    if args.gpus:
        cmd.extend(["--gpus", args.gpus])
    
    print("🧪 Running sweep test...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    # Run with output visible
    try:
        result = subprocess.run(cmd, check=False)
        
        if result.returncode != 0:
            print(f"\n❌ Sweep test failed with return code {result.returncode}")
            sys.exit(1)
        else:
            print(f"\n✅ Sweep test completed successfully!")
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running sweep test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
