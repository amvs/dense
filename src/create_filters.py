import numpy as np
import matplotlib.pyplot as plt
import torch

def gaussian(x, y, sigma_square, domain='spatial'):
    norm = 2 * np.pi * sigma_square
    if domain == 'spatial':
        value = np.exp(-(x**2 + y**2) / (2 * sigma_square))
        return value / norm
    elif domain == 'frequency':
        value = np.exp(-(x**2 + y**2) * sigma_square * 2 * np.pi**2)
        return value * norm

def cont_morlet(grid, domain, sigma=0.8, w0=0.75*np.pi, theta=0):
    sigma_square = sigma ** 2
    if domain == 'spatial':
        x, y = grid
        modulation = np.exp(1j*w0*(x*np.cos(theta) + y*np.sin(theta)))
        gau = gaussian(x, y, sigma_square)
        A = gau * modulation
        beta = np.sum(A) / np.sum(gau)
        return A - beta * gau
    elif domain == 'frequency':
        u, v = grid
        A = gaussian(u-w0*np.cos(theta), v-w0*np.sin(theta), sigma_square, domain='frequency')
        gau = gaussian(u, v, sigma_square, domain='frequency')
        A = np.fft.fftshift(A)  # Shift zero frequency component to the center
        gau = np.fft.fftshift(gau)  # Shift zero frequency component to the center
        beta = A[0,0] / (2 * np.pi * sigma_square)
        result = A - beta * gau
        return np.fft.ifftshift(result)  # Shift back to original frequency domain


def filter_bank(T, S, L):
    '''
    Create a filter bank of complex Morlet wavelets.
    T must be even, S must be odd.
    '''
    print(f'Creating filter bank with Sampling support width={T}, Size={S}, Angles={L} ...')
    filters = []
    for orient in range(L):
        theta = orient * np.pi / L
        x = np.linspace(-T//2, T//2, S) # [0, T/(S-1), 2*T/(S-1), ..., T] - T//2
        y = np.linspace(-T//2, T//2, S)
        grid = np.meshgrid(x, y)
        filter = cont_morlet(grid, domain='spatial', theta=theta)
        filter = torch.tensor(filter, dtype=torch.complex64).unsqueeze(0)  # Convert to complex tensor
        filters.append(filter)
    filters = torch.cat(filters, dim=0)  # Shape: (L, S, S)
    #torch.save(filters, './filters/morlet_S'+str(S)+'_K'+str(L)+'.pt')
    return filters

if __name__ == '__main__':
    #plot_cont_filter(10)  # Change the size as needed
    T, S, L = 8, 35, 5
    filter_bank(T, S, L)