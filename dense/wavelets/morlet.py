import numpy as np
import matplotlib.pyplot as plt


def complex_morlet_2d(scale, angle, center_frequency, S, gaussian_param=1.0):
    """
    Generate a 2D complex Morlet wavelet filter in the spatial domain.
    
    Parameters:
    -----------
    scale : float
        Scale parameter controlling the size of the wavelet
    angle : float
        Rotation angle in degrees
    center_frequency : float
        Center frequency of the wavelet (k0)
    S : int
        Size of the output array (S x S), must be odd
    gaussian_param : float, optional
        Gaussian parameter (default: 1.0)
    
    Returns:
    --------
    np.ndarray
        Complex 2D array of shape (S, S) representing the Morlet wavelet
    """
    # Ensure S is odd
    if S % 2 == 0:
        raise ValueError("S must be an odd number")
    
    # Create coordinate grid centered at origin (0, 0)
    # For odd S, center is at index (S-1)//2, which corresponds to coordinate 0
    half = (S - 1) // 2
    x = np.arange(-half, half + 1, dtype=float)
    y = np.arange(-half, half + 1, dtype=float)
    X, Y = np.meshgrid(x, y)
    
    # Convert angle to radians
    theta = np.deg2rad(angle)
    
    # Rotate coordinates
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)
    
    # Compute the distance from center (for Gaussian envelope)
    r_squared = X_rot**2 + Y_rot**2
    
    # Gaussian envelope: exp(-r^2 / (2 * sigma^2))
    # where sigma = scale * gaussian_param
    sigma = scale * gaussian_param
    gaussian_envelope = np.exp(-r_squared / (2 * sigma**2))
    
    # Complex sinusoidal component: exp(i * k0 * x_rot)
    # This creates oscillations in the direction of rotation
    # center_frequency is the wavenumber k0 (e.g., 3*pi/4), used directly
    complex_oscillation = np.exp(1j * center_frequency * X_rot)
    
    # Combine to form the complex Morlet wavelet
    wavelet = gaussian_envelope * complex_oscillation
    
    return wavelet


def filter_bank(n_scale, n_orient, S):
    """
    Generate a 2D complex Morlet wavelet filter bank.
    
    Parameters:
    -----------
    n_scale : int
        Number of scales in the filter bank
    n_orient : int
        Number of orientations in the filter bank
    S : int
        Size of each filter (S x S), must be odd
    
    Returns:
    --------
    list
        List of length n_scale, where each element is a complex array
        of shape (n_orient, S, S) containing filters at different orientations
    """
    if S % 2 == 0:
        raise ValueError("S must be an odd number")
    
    # Center frequency fixed at 3*pi/4
    center_frequency = 3 * np.pi / 4
    
    # Generate scales in geometric progression (typical for wavelet filter banks)
    # Using base 2: scales = [2^0, 2^1, 2^2, ...] or similar progression
    # Alternative: linear progression with minimum scale
    # We'll use a geometric progression starting from a base scale
    base_scale = 2.0
    scales = [base_scale * (2 ** i) for i in range(n_scale)]
    
    # Generate orientations evenly spaced from 0 to pi (not including pi)
    # This covers all unique orientations due to symmetry
    angles_rad = np.linspace(0, np.pi, n_orient, endpoint=False)
    angles_deg = np.rad2deg(angles_rad)
    
    # Initialize filter bank
    filter_bank_list = []
    
    # Generate filters for each scale
    for scale in scales:
        # Array to store filters for all orientations at this scale
        filters_at_scale = np.zeros((n_orient, S, S), dtype=complex)
        
        # Generate filter for each orientation
        for orient_idx, angle in enumerate(angles_deg):
            filter_wavelet = complex_morlet_2d(
                scale=scale,
                angle=angle,
                center_frequency=center_frequency,
                S=S,
                gaussian_param=1.0
            )
            
            # Normalize to unit L2 norm for valid wavelet filter bank
            # This ensures the filter bank satisfies the admissibility condition
            l2_norm = np.sqrt(np.sum(np.abs(filter_wavelet)**2))
            if l2_norm > 0:
                filter_wavelet = filter_wavelet / l2_norm
            
            filters_at_scale[orient_idx] = filter_wavelet
        
        filter_bank_list.append(filters_at_scale)
    
    return filter_bank_list


def plot_wavelet(wavelet, title="2D Complex Morlet Wavelet"):
    """
    Plot the real and imaginary parts of a 2D complex wavelet.
    
    Parameters:
    -----------
    wavelet : np.ndarray
        Complex 2D array representing the wavelet
    title : str, optional
        Title for the plot (default: "2D Complex Morlet Wavelet")
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Real part
    im1 = axes[0].imshow(wavelet.real, cmap='RdBu', interpolation='bilinear')
    axes[0].set_title('Real Part', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Imaginary part
    im2 = axes[1].imshow(wavelet.imag, cmap='RdBu', interpolation='bilinear')
    axes[1].set_title('Imaginary Part', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def find_optimal_crop_size(filter_wavelet, energy_threshold=0.95):
    """
    Find the optimal crop size for a filter that preserves a given percentage of energy.
    
    Parameters:
    -----------
    filter_wavelet : np.ndarray
        Complex 2D array representing the wavelet filter
    energy_threshold : float, optional
        Minimum energy percentage to preserve (default: 0.95 = 95%)
    
    Returns:
    --------
    int
        Optimal crop size N (odd number) that preserves at least energy_threshold of energy
    float
        Actual energy percentage preserved with this crop size
    """
    S = filter_wavelet.shape[0]
    center = S // 2
    
    # Calculate total energy
    total_energy = np.sum(np.abs(filter_wavelet)**2)
    
    if total_energy == 0:
        return 1, 0.0
    
    # Try different crop sizes, starting from smallest
    # Crop size must be odd
    for crop_size in range(3, S + 1, 2):  # Start from 3, step by 2 to keep odd
        half_crop = crop_size // 2
        start_idx = center - half_crop
        end_idx = center + half_crop + 1
        
        # Extract cropped region
        cropped = filter_wavelet[start_idx:end_idx, start_idx:end_idx]
        
        # Calculate energy in cropped region
        cropped_energy = np.sum(np.abs(cropped)**2)
        energy_ratio = cropped_energy / total_energy
        
        # If we've reached the threshold, return this crop size
        if energy_ratio >= energy_threshold:
            return crop_size, energy_ratio
    
    # If we couldn't reach threshold, return full size
    return S, 1.0


def crop_filter_bank(filter_bank_list, energy_threshold=0.95):
    """
    Crop a filter bank to reduce spatial domain size while preserving energy.
    Uses dyadic scaling: scale 0 uses N, scale 1 uses 2N-1, scale 2 uses 4N-1, etc.
    
    Parameters:
    -----------
    filter_bank_list : list
        List of filter arrays, each of shape (n_orient, S, S)
    energy_threshold : float, optional
        Minimum energy percentage to preserve (default: 0.95 = 95%)
    
    Returns:
    --------
    list
        Cropped filter bank, each element is of shape (n_orient, crop_size, crop_size)
    """
    n_scale = len(filter_bank_list)
    n_orient = filter_bank_list[0].shape[0]
    S = filter_bank_list[0].shape[1]
    
    # Step 1: Find optimal crop size N for scale 0
    # Use the first orientation of scale 0 as reference
    base_filter = filter_bank_list[0][0]
    base_crop_size, _ = find_optimal_crop_size(base_filter, energy_threshold)
    
    # Also check other orientations at scale 0 to ensure N is sufficient
    for orient_idx in range(1, n_orient):
        crop_size, _ = find_optimal_crop_size(filter_bank_list[0][orient_idx], energy_threshold)
        if crop_size > base_crop_size:
            base_crop_size = crop_size
    
    N = base_crop_size
    
    # Step 2: Crop all filters with dyadic scaling
    cropped_bank = []
    
    for scale_idx in range(n_scale):
        # Calculate crop size for this scale with recursive scaling:
        # Scale 0: N, Scale 1: 2N-1, Scale 2: 2(2N-1)-1, Scale 3: 2(2(2N-1)-1)-1, etc.
        # Formula: current = 2 * previous - 1
        if scale_idx == 0:
            scale_crop_size = N
        else:
            # Calculate recursively: start from N and apply the formula scale_idx times
            scale_crop_size = N
            for i in range(scale_idx):
                scale_crop_size = 2 * scale_crop_size - 1
        # Make it odd if needed
        if scale_crop_size % 2 == 0:
            scale_crop_size += 1
        # Don't exceed original size
        scale_crop_size = min(scale_crop_size, S)
        # Ensure minimum size of 3
        scale_crop_size = max(scale_crop_size, 3)
        
        # Crop filters at this scale
        center = S // 2
        half_crop = scale_crop_size // 2
        start_idx = center - half_crop
        end_idx = center + half_crop + 1
        
        cropped_filters = np.zeros((n_orient, scale_crop_size, scale_crop_size), dtype=complex)
        
        for orient_idx in range(n_orient):
            original_filter = filter_bank_list[scale_idx][orient_idx]
            # Crop the filter
            cropped_filter = original_filter[start_idx:end_idx, start_idx:end_idx]
            cropped_filters[orient_idx] = cropped_filter
        
        cropped_bank.append(cropped_filters)
    
    return cropped_bank


def calculate_crop_statistics(original_bank, cropped_bank, energy_threshold=0.95):
    """
    Calculate statistics about the energy retention after cropping.
    
    Parameters:
    -----------
    original_bank : list
        Original filter bank, each element of shape (n_orient, S, S)
    cropped_bank : list
        Cropped filter bank, each element of shape (n_orient, crop_size, crop_size)
    energy_threshold : float, optional
        Energy threshold used for cropping (default: 0.95)
    
    Returns:
    --------
    dict
        Statistics dictionary containing:
        - 'base_crop_size': N (crop size for scale 0)
        - 'crop_sizes': list of crop sizes for each scale
        - 'energy_retained': list of energy percentages for each scale/orientation
        - 'avg_energy_retained': average energy retained per scale
        - 'min_energy_retained': minimum energy retained per scale
        - 'max_energy_retained': maximum energy retained per scale
    """
    n_scale = len(original_bank)
    n_orient = original_bank[0].shape[0]
    S = original_bank[0].shape[1]
    
    crop_sizes = []
    energy_retained = []
    
    # Get base crop size from scale 0
    base_crop_size = cropped_bank[0].shape[1]
    
    for scale_idx in range(n_scale):
        crop_size = cropped_bank[scale_idx].shape[1]
        crop_sizes.append(crop_size)
        
        scale_energies = []
        for orient_idx in range(n_orient):
            original_filter = original_bank[scale_idx][orient_idx]
            cropped_filter = cropped_bank[scale_idx][orient_idx]
            
            # Calculate original energy
            original_energy = np.sum(np.abs(original_filter)**2)
            
            # Calculate energy in cropped filter
            cropped_energy = np.sum(np.abs(cropped_filter)**2)
            
            # Calculate energy retention percentage
            if original_energy > 0:
                energy_ratio = cropped_energy / original_energy
            else:
                energy_ratio = 1.0
            
            scale_energies.append(energy_ratio)
        
        energy_retained.append(scale_energies)
    
    # Calculate statistics
    avg_energy_per_scale = [np.mean(energies) for energies in energy_retained]
    
    stats = {
        'base_crop_size': base_crop_size,
        'crop_sizes': crop_sizes,
        'energy_retained': energy_retained,
        'avg_energy_retained': avg_energy_per_scale,
        'min_energy_retained': [np.min(energies) for energies in energy_retained],
        'max_energy_retained': [np.max(energies) for energies in energy_retained],
        'energy_threshold': energy_threshold,
        'original_size': S
    }
    
    return stats


def print_crop_statistics(stats):
    """
    Print crop statistics in a formatted way.
    
    Parameters:
    -----------
    stats : dict
        Statistics dictionary from calculate_crop_statistics
    """
    n_scale = len(stats['crop_sizes'])
    n_orient = len(stats['energy_retained'][0])
    S = stats['original_size']
    N = stats['base_crop_size']
    energy_threshold = stats['energy_threshold']
    
    print(f"\nFilter Bank Cropping Statistics:")
    print(f"  Base crop size N (scale 0): {N}")
    print(f"  Energy threshold: {energy_threshold*100:.1f}%")
    print(f"\n  Crop sizes per scale:")
    for scale_idx in range(n_scale):
        crop_size = stats['crop_sizes'][scale_idx]
        print(f"    Scale {scale_idx}: {crop_size}x{crop_size} "
              f"(original: {S}x{S}, reduction: {(1 - crop_size/S)*100:.1f}%)")
    
    print(f"\n  Energy retention per scale:")
    for scale_idx in range(n_scale):
        avg = stats['avg_energy_retained'][scale_idx]
        min_e = stats['min_energy_retained'][scale_idx]
        max_e = stats['max_energy_retained'][scale_idx]
        print(f"    Scale {scale_idx}:")
        print(f"      Average: {avg*100:.2f}%")
        print(f"      Range: {min_e*100:.2f}% - {max_e*100:.2f}%")
        print(f"      Per orientation:")
        for orient_idx in range(n_orient):
            energy_pct = stats['energy_retained'][scale_idx][orient_idx] * 100
            print(f"        Orientation {orient_idx}: {energy_pct:.2f}%")


def plot_wavelet_spatial_frequency(wavelet, title="2D Complex Morlet Wavelet"):
    """
    Plot the wavelet in both spatial domain and frequency domain (using FFT).
    
    Parameters:
    -----------
    wavelet : np.ndarray
        Complex 2D array representing the wavelet
    title : str, optional
        Title for the plot (default: "2D Complex Morlet Wavelet")
    """
    # Compute FFT and shift zero frequency to center
    fft_wavelet = np.fft.fftshift(np.fft.fft2(wavelet))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # Spatial domain - Real part
    im1 = axes[0, 0].imshow(wavelet.real, cmap='RdBu', interpolation='bilinear')
    axes[0, 0].set_title('Spatial Domain - Real Part', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # Spatial domain - Imaginary part
    im2 = axes[0, 1].imshow(wavelet.imag, cmap='RdBu', interpolation='bilinear')
    axes[0, 1].set_title('Spatial Domain - Imaginary Part', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Frequency domain - Magnitude
    magnitude = np.abs(fft_wavelet)
    im3 = axes[1, 0].imshow(magnitude, cmap='hot', interpolation='bilinear')
    axes[1, 0].set_title('Frequency Domain - Magnitude', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Frequency domain - Phase
    phase = np.angle(fft_wavelet)
    im4 = axes[1, 1].imshow(phase, cmap='hsv', interpolation='bilinear')
    axes[1, 1].set_title('Frequency Domain - Phase', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def morlet(n_scale, n_orient, S=129):
    cropped_bank = []
    bank = filter_bank(n_scale=n_scale, n_orient=n_orient, S=S)
    cropped_bank = crop_filter_bank(bank, energy_threshold=0.95)
    return cropped_bank

if __name__ == "__main__":
    # Example 1: Small scale, 0 degrees, low frequency
    print("Example 1: Small scale, 0 degrees, low frequency")
    wavelet1 = complex_morlet_2d(scale=5, angle=0, center_frequency=0.1, S=129)
    plot_wavelet(wavelet1, "Example 1: scale=5, angle=0°, freq=0.1")
    
    # Example 2: Medium scale, 45 degrees, medium frequency
    print("Example 2: Medium scale, 45 degrees, medium frequency")
    wavelet2 = complex_morlet_2d(scale=10, angle=45, center_frequency=0.3, S=129)
    plot_wavelet(wavelet2, "Example 2: scale=10, angle=45°, freq=0.3")
    
    # Example 3: Large scale, 90 degrees, high frequency
    print("Example 3: Large scale, 90 degrees, high frequency")
    wavelet3 = complex_morlet_2d(scale=15, angle=90, center_frequency=0.5, S=129)
    plot_wavelet(wavelet3, "Example 3: scale=15, angle=90°, freq=0.5")
    
    # Example 4: Different angle and frequency
    print("Example 4: 30 degrees, medium frequency")
    wavelet4 = complex_morlet_2d(scale=8, angle=30, center_frequency=0.2, S=129)
    plot_wavelet(wavelet4, "Example 4: scale=8, angle=30°, freq=0.2")
    
    # Print some statistics
    print("\nWavelet Statistics:")
    print(f"Example 1 - Max real: {wavelet1.real.max():.4f}, Max imag: {wavelet1.imag.max():.4f}")
    print(f"Example 2 - Max real: {wavelet2.real.max():.4f}, Max imag: {wavelet2.imag.max():.4f}")
    print(f"Example 3 - Max real: {wavelet3.real.max():.4f}, Max imag: {wavelet3.imag.max():.4f}")
    print(f"Example 4 - Max real: {wavelet4.real.max():.4f}, Max imag: {wavelet4.imag.max():.4f}")
    
    # Test filter_bank function
    print("\n" + "="*60)
    print("Testing filter_bank function")
    print("="*60)
    
    # Create a filter bank with 3 scales and 3 orientations
    n_scale = 3
    n_orient = 3
    S = 65  # Odd number
    
    print(f"\nCreating filter bank: n_scale={n_scale}, n_orient={n_orient}, S={S}")
    bank = filter_bank(n_scale=n_scale, n_orient=n_orient, S=S)
    
    print(f"\nFilter bank structure:")
    print(f"  - Number of scales: {len(bank)}")
    base_scale = 2.0
    for i, filters_at_scale in enumerate(bank):
        scale_val = base_scale * (2 ** i)
        print(f"  - Scale {i}: shape {filters_at_scale.shape}")
        print(f"    - Scale value: {scale_val:.2f}")
        print(f"    - Orientations: {n_orient} (0° to {180*(n_orient-1)/n_orient:.1f}°)")
        # Verify L2 normalization
        for j in range(n_orient):
            l2_norm = np.sqrt(np.sum(np.abs(filters_at_scale[j])**2))
            print(f"      - Filter [{i}, {j}] L2 norm: {l2_norm:.6f}")
    
    # Plot all filters in spatial and frequency domains
    print("\nPlotting all filters from filter bank (spatial and frequency domains):")
    
    for scale_idx in range(n_scale):
        for orient_idx in range(n_orient):
            scale_val = base_scale * (2 ** scale_idx)
            angle_deg = orient_idx * 180 / n_orient
            title = f"Filter Bank: Scale {scale_idx} (σ={scale_val:.1f}), Orientation {orient_idx} ({angle_deg:.1f}°)"
            print(f"  - Scale {scale_idx}, Orientation {orient_idx}")
            plot_wavelet_spatial_frequency(bank[scale_idx][orient_idx], title)
    
    # Test filter bank cropping
    print("\n" + "="*60)
    print("Testing filter bank cropping")
    print("="*60)
    
    # Crop the filter bank
    cropped_bank = crop_filter_bank(bank, energy_threshold=0.95)
    
    # Calculate and print statistics
    stats = calculate_crop_statistics(bank, cropped_bank, energy_threshold=0.95)
    print_crop_statistics(stats)
    
    # Compare original vs cropped sizes
    print(f"\nSize comparison:")
    print(f"  Original filter size: {S}x{S} = {S*S} pixels")
    total_original = sum(S*S for _ in range(n_scale))
    total_cropped = sum(size*size for size in stats['crop_sizes'])
    print(f"  Total pixels (all scales, all orientations):")
    print(f"    Original: {total_original * n_orient} pixels")
    print(f"    Cropped:  {total_cropped * n_orient} pixels")
    print(f"    Reduction: {(1 - total_cropped/total_original)*100:.1f}%")
    
    # Plot comparison: original vs cropped for one filter
    print(f"\nPlotting comparison: Original vs Cropped (Scale 0, Orientation 0):")
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # Original - Real
    im1 = axes[0, 0].imshow(bank[0][0].real, cmap='RdBu', interpolation='bilinear')
    axes[0, 0].set_title(f'Original - Real Part ({S}x{S})', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # Cropped - Real
    crop_size_0 = stats['crop_sizes'][0]
    im2 = axes[0, 1].imshow(cropped_bank[0][0].real, cmap='RdBu', interpolation='bilinear')
    axes[0, 1].set_title(f'Cropped - Real Part ({crop_size_0}x{crop_size_0})', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Original - Imaginary
    im3 = axes[1, 0].imshow(bank[0][0].imag, cmap='RdBu', interpolation='bilinear')
    axes[1, 0].set_title(f'Original - Imaginary Part ({S}x{S})', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Cropped - Imaginary
    im4 = axes[1, 1].imshow(cropped_bank[0][0].imag, cmap='RdBu', interpolation='bilinear')
    axes[1, 1].set_title(f'Cropped - Imaginary Part ({crop_size_0}x{crop_size_0})', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    energy_pct = stats['energy_retained'][0][0] * 100
    plt.suptitle(f'Original vs Cropped Filter (Energy Retained: {energy_pct:.2f}%)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()