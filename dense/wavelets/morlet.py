import torch
import numpy as np
import logging

def gaussian_2d(x, y, sigma):
    """
    Computes a normalized 2D Gaussian.
    Expects 'sigma' (std dev), NOT sigma squared.
    """
    sigma_sq = sigma**2
    norm = 2 * np.pi * sigma_sq
    term = -(x**2 + y**2) / (2 * sigma_sq)
    return np.exp(term) / norm

def cont_morlet(grid, theta, sigma, xi):
    """
    Generates a Morlet wavelet in the Spatial Domain.
    """
    x, y = grid
    
    # 1. Rotate coordinates
    u_prime = x * np.cos(theta) + y * np.sin(theta)
    
    # 2. Compute Envelope (Gaussian)
    # FIX: Pass sigma directly (gaussian_2d handles the squaring)
    gau = gaussian_2d(x, y, sigma)
    
    # 3. Compute Oscillation
    modulation = np.exp(1j * xi * u_prime)
    
    # 4. Raw Morlet
    A = gau * modulation
    
    # 5. Admissibility Correction
    beta = np.sum(A) / np.sum(gau)
    psi = A - beta * gau
    
    return psi

def filter_bank_safe(S, L, sigma, xi=np.pi*3/4):
    """
    Creates a single bank of filters for a specific scale (sigma) and size (S).
    """
    # Grid setup: Integers [-S//2 ... S//2]
    range_val = S // 2
    x = np.linspace(-range_val, range_val, S)
    y = np.linspace(-range_val, range_val, S)
    grid = np.meshgrid(x, y)
    
    # Safety Checks
    if S < 4 * sigma:
        logging.warning(f"S={S} is too small for sigma={sigma:.2f} (Truncation)")

    filters = []
    for orient in range(L):
        theta = orient * np.pi / L
        
        # Generate filter
        filter_val = cont_morlet(grid, theta, sigma, xi)
        
        # L1 Normalization
        filter_val /= np.abs(filter_val).sum()
        
        filters.append(torch.tensor(filter_val, dtype=torch.complex64).unsqueeze(0))
        
    return torch.cat(filters, dim=0)

def morlet(max_scale, nb_orients, S=7, sigma=1.0, xi=np.pi*3/4):
    """
    Creates a multi-scale filter bank.
    Correctly dilates sigma and xi for each scale j.
    """
    filters = []
    
    current_S = S
    current_sigma = sigma
    current_xi = xi
    
    for j in range(max_scale):
        # Create bank for this scale
        print(f"Scale {j}: S={current_S}, Sigma={current_sigma:.2f}, Freq={current_xi:.2f}")
        
        f_bank = filter_bank_safe(S=current_S, L=nb_orients, sigma=current_sigma, xi=current_xi)
        filters.append(f_bank)
        
        # UPDATE SCALES FOR NEXT LAYER (Dilation)
        # 1. Spatial support grows roughly 2x (or just add padding)
        current_S = 2 * current_S + 1 
        
        # 2. Envelope dilates 2x
        current_sigma *= 2.0
        
        # 3. Frequency drops 2x
        current_xi /= 2.0
        
    return filters