"""
Copied from wavelet-texture-synthesis repo: https://github.com/abrochar/wavelet-texture-synthesis/
Original author: Antoine Brochard
"""
import numpy as np
import torch
from kymatio.scattering2d.filter_bank import filter_bank
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=256)
parser.add_argument('--J', type=int, default=5)
parser.add_argument('--L', type=int, default=4)
parser.add_argument('--wavelets', default='morlet')
args = parser.parse_args()

N = args.N
J = args.J
L = args.L

if args.wavelets == 'morlet':
    dict_hl = filter_bank(N, N, J, L)
    # high-pass filter
    dict_psi = dict_hl['psi']
    hatpsi = torch.FloatTensor(J, L, N, N, 2)
    for j in range(J):
        for theta in range(L):
            hatpsi[j,theta,:,:,:] = torch.stack((torch.Tensor(np.real(dict_psi[L*j+theta]['levels'][0])),
                                                torch.Tensor(np.imag(dict_psi[L*j+theta]['levels'][0]).copy())),
                                               dim=-1)
    hatpsi = torch.view_as_complex(hatpsi)
    torch.save(hatpsi, './filters/morlet_N'+str(N)+'_J'+str(J)+'_L'+str(L)+'.pt')
    print('saved filters, starting low pass filter')
    # low-pass filter
    dict_phi = dict_hl['phi']
    hatphi = torch.stack((torch.Tensor(np.real(dict_phi['levels'][0])),
                          torch.Tensor(np.imag(dict_phi['levels'][0]).copy())),
                         dim=-1)
    hatphi = torch.view_as_complex(hatphi)
    torch.save(hatphi, './filters/morlet_lp_N'+str(N)+'_J'+str(J)+'_L'+str(L)+'.pt')
    print('saved low pass filter, done')




