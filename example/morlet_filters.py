# import numpy as np
# from scipy.fft import fft2, ifft2
# import torch


# def morlet2d(x, y, theta, sigma=0.25, w0=0.75, scale=1): #0.8, 3/4pi
#     x = x / scale
#     y = y / scale
#     norm = 2 * 3.1415 * sigma * sigma
#     value = np.exp(-(x**2 + y**2)/(2*sigma**2)) * np.exp(1j*w0*(x*np.cos(theta)+y*np.sin(theta))) 
#     return value / norm

# def filter_bank(S, L):
#     sample_interval = 1.35
#     temp = []
#     for orient in range(L):
#         theta = orient * np.pi / L
#         x = np.linspace(-(S//2), S//2, S) * sample_interval
#         y = np.linspace(-(S//2), S//2, S) * sample_interval
#         X, Y = np.meshgrid(x, y)
#         filter = morlet2d(X, Y, theta)
#         filter = torch.tensor(filter, dtype=torch.complex64).unsqueeze(0) # 1,1,s,s
#         temp.append(filter)
#     psi = torch.cat(temp, dim=0) # orient, s, s
#     torch.save(psi, './filters/morlet_S'+str(S)+'_K'+str(L)+'.pt')


# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--L', type=int, default=16)
# parser.add_argument('--S', type=int, default=3)
# args = parser.parse_args()

# L = args.L
# S = args.S
# filter_bank(S, L)

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


def plot_cont_filter(N):
    x = np.linspace(-N//2, N//2, 1000)
    y = np.linspace(-N//2, N//2, 1000)
    grid = np.meshgrid(x, y)
    spatial_z = cont_morlet(grid, domain='spatial')
    freq_z = cont_morlet(grid, domain='frequency')

    real_spatial = spatial_z.real.copy()
    epsilon = 1e-5
    real_spatial[real_spatial < -epsilon] = -1
    real_spatial[real_spatial > epsilon] = 1

    imag_spatial = spatial_z.imag.copy()
    imag_spatial[imag_spatial < -epsilon] = -1
    imag_spatial[imag_spatial > epsilon] = 1

    real_freq = freq_z.real.copy()
    print(real_freq[0, 0])
    real_freq[real_freq < -epsilon] = -1
    real_freq[real_freq > epsilon] = 1

    imag_freq = freq_z.imag.copy()
    imag_freq[imag_freq < -epsilon] = -1
    imag_freq[imag_freq > epsilon] = 1

    # Plot the image
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(221)
    ax1.imshow(real_spatial, cmap='gray', extent=(-N//2, N//2, -N//2, N//2))
    ax1.set_title('Real Part of Spatial Domain')
    ax1.set_xticks(np.arange(-N//2, N//2+1, 1))
    ax1.set_yticks(np.arange(-N//2, N//2+1, 1))
    ax1.grid(True)


    ax2 = fig.add_subplot(222)
    ax2.imshow(imag_spatial, cmap='gray', extent=(-N//2, N//2, -N//2, N//2))
    ax2.set_title('Imaginary Part of Spatial Domain')
    ax2.set_xticks(np.arange(-N//2, N//2+1, 1))
    ax2.set_yticks(np.arange(-N//2, N//2+1, 1))
    ax2.grid(True)  

    ax3 = fig.add_subplot(223)
    ax3.imshow(real_freq, cmap='gray', extent=(-N//2, N//2, -N//2, N//2))
    ax3.set_title('Real Part of Frequency Domain')
    ax3.set_xticks(np.arange(-N//2, N//2+1, 1))
    ax3.set_yticks(np.arange(-N//2, N//2+1, 1))
    ax3.grid(True)

    ax4 = fig.add_subplot(224)
    ax4.imshow(imag_freq, cmap='gray', extent=(-N//2, N//2, -N//2, N//2))
    ax4.set_title('Imaginary Part of Frequency Domain')
    ax4.set_xticks(np.arange(-N//2, N//2+1, 1))
    ax4.set_yticks(np.arange(-N//2, N//2+1, 1))
    ax4.grid(True)
    plt.tight_layout()
    plt.show()



def plot_surface_complex(data, title):
    fig = plt.figure(figsize=(12, 6))
    
    # Plot spatial domain
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(grid[0], grid[1], data.real, cmap='viridis')
    ax1.set_title(title + '(Real Part)')
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    ax1.set_zlabel('Amplitude')

    ax1 = fig.add_subplot(122, projection='3d')
    ax1.plot_surface(grid[0], grid[1], data.imag, cmap='viridis')
    ax1.set_title(title + '(Imag Part)')
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    ax1.set_zlabel('Amplitude')

    plt.tight_layout()
    plt.show()

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


def plot_filter_bank(filter, T):
    freq_filter = np.fft.fft2(filter)
    freq_filter = np.fft.fftshift(freq_filter)  # Shift zero frequency component

    real_filter = filter.real.copy()
    epsilon = 1e-5
    real_filter[real_filter < -epsilon] = -1
    real_filter[real_filter > epsilon] = 1
    imag_filter = filter.imag.copy()
    imag_filter[imag_filter < -epsilon] = -1
    imag_filter[imag_filter > epsilon] = 1  
    real_freq_filter = freq_filter.real.copy()
    real_freq_filter[real_freq_filter < -epsilon] = -1
    real_freq_filter[real_freq_filter > epsilon] = 1
    imag_freq_filter = freq_filter.imag.copy()
    imag_freq_filter[imag_freq_filter < -epsilon] = -1
    imag_freq_filter[imag_freq_filter > epsilon] = 1

    # Plot the filters
    fig = plt.figure(figsize=(24, 24))
    ax1 = fig.add_subplot(221)
    ax1.imshow(real_filter, cmap='gray', extent=(-T//2, T//2, -T//2, T//2))
    ax1.set_title('Spatial Domain Filter')
    ax1.set_xticks(np.arange(-T//2, T//2+1, 1))
    ax1.set_yticks(np.arange(-T//2, T//2+1, 1))
    ax1.grid(True)

    ax2 = fig.add_subplot(222)
    ax2.imshow(imag_filter, cmap='gray', extent=(-T//2, T//2, -T//2, T//2))
    ax2.set_title('Imaginary Part of Spatial Domain Filter')
    ax2.set_xticks(np.arange(-T//2, T//2+1, 1))
    ax2.set_yticks(np.arange(-T//2, T//2+1, 1))
    ax2.grid(True)

    ax3 = fig.add_subplot(223)
    ax3.imshow(real_freq_filter, cmap='gray', extent=(-T//2, T//2, -T//2, T//2))
    ax3.set_title('Frequency Domain Filter')
    ax3.set_xticks(np.arange(-T//2, T//2+1, 1))
    ax3.grid(True)

    ax4 = fig.add_subplot(224)
    ax4.imshow(imag_freq_filter, cmap='gray', extent=(-T//2, T//2, -T//2, T//2))
    ax4.set_title('Imaginary Part of Frequency Domain Filter')
    ax4.set_xticks(np.arange(-T//2, T//2+1, 1))
    ax4.set_yticks(np.arange(-T//2, T//2+1, 1))
    ax4.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # plot_cont_filter(10)  # Change the size as needed
    T, S, L = 8, 35, 5
    filters = np.array(filter_bank(T, S, L))
    plot_filter_bank(filters[0], T)  # Plot the first filter in the bank