"""
Standalone test script to inspect get_feature_metadata() logic.
This version tests the metadata computation logic directly without full model initialization.
"""
import torch

def test_metadata_logic(dmax):
    """
    Test the metadata computation logic directly.
    This mimics what get_feature_metadata() does in the dense class.
    """
    # Test parameters
    dmax = J if dmax == -1 else dmax
    
    print("=" * 80)
    print("Feature Metadata Logic Test")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  n_scale (J): {J}")
    print(f"  n_orient (K): {K}")
    print(f"  depth (dmax): {dmax}")
    
    # Build metadata by tracking paths through the scattering tree
    # Track: (depth, scales_path, angles_path) for each input group
    input_paths = [(0, [], [])]  # Start with depth=0 input image
    
    all_metadata = []
    
    # Depth 0: input image
    all_metadata.append((0, [], []))
    
    print("\n" + "=" * 80)
    print("Building metadata tree:")
    print("=" * 80)
    
    for scale in range(J):
        output_paths = []
        
        print(f"\nScale {scale}:")
        print(f"  Input paths: {input_paths}")
        
        # Process each input path
        for dep, scales_path, angles_path in input_paths:
            if dep >= dmax:
                continue
            
            # Each input path produces K output paths (one per orientation)
            for angle in range(K):
                new_depth = dep + 1
                new_scales = scales_path + [scale]
                new_angles = angles_path + [angle]
                output_paths.append((new_depth, new_scales, new_angles))
                all_metadata.append((new_depth, new_scales, new_angles))
                print(f"    Input (depth={dep}, scales={scales_path}, angles={angles_path}) -> "
                      f"Output (depth={new_depth}, scales={new_scales}, angles={new_angles})")
        
        # Update input paths for next scale
        if scale < J - 1:
            # Keep all previous paths (they continue as low-pass)
            input_paths = [
                (dep, scales, angles) 
                for dep, scales, angles in input_paths 
                if dep < dmax
            ]
            # Add new output paths
            input_paths.extend(output_paths)
            print(f"  Updated input_paths for next scale: {len(input_paths)} paths")
    
    # Convert to tensor format [depth, scale_1, angle_1, scale_2, angle_2]
    metadata_tensor = torch.zeros(len(all_metadata), 5, dtype=torch.long)
    for idx, (depth, scales, angles) in enumerate(all_metadata):
        metadata_tensor[idx, 0] = depth
        if len(scales) >= 1:
            metadata_tensor[idx, 1] = scales[0]
            metadata_tensor[idx, 2] = angles[0]
        else:
            metadata_tensor[idx, 1] = -1
            metadata_tensor[idx, 2] = -1
        if len(scales) >= 2:
            metadata_tensor[idx, 3] = scales[1]
            metadata_tensor[idx, 4] = angles[1]
        else:
            metadata_tensor[idx, 3] = -1
            metadata_tensor[idx, 4] = -1
    
    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"\nTotal feature maps: {len(all_metadata)}")
    print(f"Metadata tensor shape: {metadata_tensor.shape}")
    print(f"\nMetadata format: [depth, scale_1, angle_1, scale_2, angle_2]")
    
    print("\n" + "=" * 80)
    print("First 30 feature maps:")
    print("=" * 80)
    print(f"{'Index':<8} {'Depth':<8} {'Scale1':<8} {'Angle1':<8} {'Scale2':<8} {'Angle2':<8} {'Path':<30}")
    print("-" * 100)
    
    for i in range(min(30, len(metadata_tensor))):
        row = metadata_tensor[i]
        depth = row[0].item()
        scale1 = row[1].item()
        angle1 = row[2].item()
        scale2 = row[3].item()
        angle2 = row[4].item()
        
        # Reconstruct path description
        if depth == 0:
            path_str = "input"
        elif depth == 1:
            path_str = f"s{scale1}_a{angle1}"
        else:
            path_str = f"s{scale1}_a{angle1}_s{scale2}_a{angle2}"
        
        print(f"{i:<8} {depth:<8} {scale1:<8} {angle1:<8} {scale2:<8} {angle2:<8} {path_str:<30}")
    
    if len(metadata_tensor) > 30:
        print(f"\n... (showing first 30 of {len(metadata_tensor)} total)")
    
    print("\n" + "=" * 80)
    print("Statistics:")
    print("=" * 80)
    
    # Count features by depth
    depths = metadata_tensor[:, 0]
    unique_depths, counts = torch.unique(depths, return_counts=True)
    print("\nFeatures by depth:")
    for depth, count in zip(unique_depths, counts):
        print(f"  Depth {depth.item()}: {count.item()} feature maps")
    
    # Expected counts
    print("\nExpected counts:")
    print(f"  Depth 0: 1 (input image)")
    print(f"  Depth 1: {J * K} (J scales × K orientations)")
    expected_depth2 = sum([(i+1) * K * K for i in range(J)]) if J > 1 else 0
    print(f"  Depth 2: {expected_depth2} (for J={J}, K={K})")
    
    print("\n" + "=" * 80)
    print("Full metadata tensor:")
    print("=" * 80)
    print(metadata_tensor)
    
    return metadata_tensor

if __name__ == "__main__":
    J = 3  # n_scale
    K = 2  # n_orient
    dmax = 2
    metadata = test_metadata_logic(dmax)
