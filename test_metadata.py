"""
Minimal test script to inspect get_feature_metadata() output.
"""
import sys
import os
# Add the current directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
# Import directly to avoid logger dependencies
from dense.model import dense, ScatterParams

def test_metadata():
    # Create a simple test configuration
    params = ScatterParams(
        n_scale=3,           # 3 scales
        n_orient=4,         # 4 orientations
        n_copies=1,
        in_channels=1,
        wavelet="morlet",
        n_class=10,
        share_channels=False,
        in_size=32,         # 32x32 input image
        depth=-1,           # Full scatter (no depth limit)
        random=False,
        classifier_type='hypernetwork'
    )
    
    # Create model
    model = dense(params)
    
    # Get metadata
    metadata = model.get_feature_metadata()
    
    print("=" * 80)
    print("Feature Metadata Test")
    print("=" * 80)
    print(f"\nModel configuration:")
    print(f"  n_scale: {params.n_scale}")
    print(f"  n_orient: {params.n_orient}")
    print(f"  depth: {params.depth} (full scatter)")
    print(f"  Expected total features: {model.out_channels}")
    
    print(f"\nMetadata shape: {metadata.shape}")
    print(f"Number of feature maps: {metadata.shape[0]}")
    print(f"\nMetadata format: [depth, scale_1, angle_1, scale_2, angle_2]")
    print("\n" + "=" * 80)
    print("First 20 feature maps:")
    print("=" * 80)
    print(f"{'Index':<8} {'Depth':<8} {'Scale1':<8} {'Angle1':<8} {'Scale2':<8} {'Angle2':<8}")
    print("-" * 80)
    
    for i in range(min(20, len(metadata))):
        row = metadata[i]
        print(f"{i:<8} {row[0].item():<8} {row[1].item():<8} {row[2].item():<8} {row[3].item():<8} {row[4].item():<8}")
    
    if len(metadata) > 20:
        print(f"\n... (showing first 20 of {len(metadata)} total)")
    
    print("\n" + "=" * 80)
    print("Statistics:")
    print("=" * 80)
    
    # Count features by depth
    depths = metadata[:, 0]
    unique_depths, counts = torch.unique(depths, return_counts=True)
    print("\nFeatures by depth:")
    for depth, count in zip(unique_depths, counts):
        print(f"  Depth {depth.item()}: {count.item()} feature maps")
    
    # Show some examples of different depths
    print("\n" + "=" * 80)
    print("Examples by depth:")
    print("=" * 80)
    
    for depth_val in unique_depths:
        depth_val = depth_val.item()
        indices = (depths == depth_val).nonzero(as_tuple=True)[0]
        if len(indices) > 0:
            idx = indices[0].item()
            row = metadata[idx]
            print(f"\nDepth {depth_val} example (index {idx}):")
            print(f"  [depth={row[0].item()}, scale_1={row[1].item()}, angle_1={row[2].item()}, "
                  f"scale_2={row[3].item()}, angle_2={row[4].item()}]")
            if len(indices) > 1:
                idx2 = indices[min(3, len(indices)-1)].item()
                row2 = metadata[idx2]
                print(f"  [depth={row2[0].item()}, scale_1={row2[1].item()}, angle_1={row2[2].item()}, "
                      f"scale_2={row2[3].item()}, angle_2={row2[4].item()}]")
    
    print("\n" + "=" * 80)
    print("Full metadata tensor:")
    print("=" * 80)
    print(metadata)
    
    return metadata

if __name__ == "__main__":
    metadata = test_metadata()
