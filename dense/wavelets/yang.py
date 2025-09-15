import numpy as np
import matplotlib.pyplot as plt
import torch
from dense.helpers import LoggerManager

def radial(omega):
        '''
            Evaluate frequency function \hat{q}(\omega)

            when -1 < \omega < 1, \hat{q}(w) = cos(w*pi/2)
        '''
        return np.cos(np.pi * omega / 2) * ((omega > -1) & (omega < 1))

def angular(omega, nb_orients):
        '''
            \hat{y} is 2*pi periodic function
            let \theta = pi/nb_orients or 2*pi/nb_orients
            when -\theta < w < \theta, \hat{y}(w) = cos(K*w/4)
        '''
        theta = np.pi/nb_orients
        omega = np.where(omega > np.pi, omega - 2 * np.pi, omega)
        return np.cos(nb_orients * omega / 2) * ((omega > -theta ) & (omega < theta ))


def filter_bank(j, nb_orients, x, y, P):
    '''
    Create a filter bank of complex Yang wavelets at scale j.
    '''
    filters = []
    for k in range(nb_orients):
        theta_k = k * np.pi / nb_orients
        periodize_freq_signal = np.zeros((P, P))
        for xx in [-1, 0, 1]:
            for yy in [-1, 0, 1]:
                x_shifted = x + xx * np.pi
                y_shifted = y + yy * np.pi
                R = np.sqrt(x_shifted**2 + y_shifted**2)
                Theta = np.atan2(x_shifted, y_shifted)
                radial_input = np.log2(R + 1e-10)
                angular_input = Theta
                freq_signal = radial(radial_input + j) * angular(angular_input - theta_k, nb_orients)
                periodize_freq_signal += freq_signal
        shifted_signal = np.fft.ifftshift(periodize_freq_signal, axes=(-2,-1))  # Shift zero frequency component to the center
        kernel = np.fft.ifft2(shifted_signal)
        centered_filter = np.fft.fftshift(kernel, axes=(-2,-1))  # Center the filter in spatial domain
        filter = torch.tensor(centered_filter, dtype=torch.complex64).unsqueeze(0)  # Convert to complex tensor
        filters.append(filter)
        # plt.imshow(shifted_signal)
        # plt.show()
        # plt.imshow(centered_filter.real)
        # plt.show()
        # plt.imshow(centered_filter.imag)
        # plt.show()
        # plt.imsave(f'{j=}_{k=}_freq.png', shifted_signal)
        # plt.imsave(f'{j=}_{k=}_spatial_real.png', centered_filter.real)
        # plt.imsave(f'{j=}_{k=}_spatial_imag.png', centered_filter.imag)
    filters = torch.cat(filters, dim=0)  # Shape: (nb_orients, P, P)

    return filters 


def yang(max_scale, nb_orients, P=256, S=7):
    '''
        S = 13 is the best recommended size

    '''
    if S%2==0:
        raise ValueError("[Yang filter]: S={S} must be odd.") 
    logger = LoggerManager.get_logger()
    logger.info(f'Creating filter bank with Sampling support Size={S}, Angles={nb_orients}, Scales={max_scale} ...')

    k = np.fft.fftfreq(P, d=2.0) * 2 * np.pi
    x, y = np.meshgrid(k, k, indexing='ij')
    filters = []
    for j in range(max_scale):
        filter = filter_bank(j, nb_orients, x, y, P)
        # crop to size SxS
        start = (P - S) // 2
        filter = filter[:, start:start+S, start:start+S]
        filters.append(filter)
        S = 2*S-1
    return filters

if __name__ == "__main__":
    max_scale = 4
    nb_orients = 6
    P = 256
    filters = yang(max_scale, nb_orients, P)
    