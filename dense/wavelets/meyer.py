import numpy as np
import torch
import matplotlib.pyplot as plt
PI = np.pi
NYQUIST_T = 3/8
PERIOD = 2*PI


def smoothing(p):
    t = np.clip(p, 0.0, 1.0)
    return t**4 * (35 - 84*t + 70*t**2 - 20*t**3)


def radial(rho):
    """
    Classic radial Meyer band: support (2pi/3, 8pi/3)
    """
    out = np.zeros_like(rho)
    m1 = (rho > 2*PI/3) & (rho < 4*PI/3)
    m2 = (rho >= 4*PI/3) & (rho < 8*PI/3)
    out[m1] = np.sin((PI/2) * smoothing(-1 + 3*rho[m1]/(2*PI)))
    out[m2] = np.cos((PI/2) * smoothing(-1 + 3*rho[m2]/(4*PI)))
    return out


def angular_window(theta, K, k, eps=1e-12):
    """
    Angular window for Partition of unity
    """
    phi = np.mod(theta, 2*PI)
    r = (K/(2*PI)) * phi

    u1 = r - k
    u2 = r - ((k - 1) % K)

    m1 = (u1 >= 0.0) & (u1 < 1.0 - eps)
    m2 = (u2 >= 0.0) & (u2 < 1.0 - eps)

    s1 = smoothing(np.clip(u1, 0.0, 1.0 - eps))
    s2 = smoothing(np.clip(u2, 0.0, 1.0 - eps))

    out = np.zeros_like(phi, float)
    out[m1] = np.cos(0.5*PI * s1[m1])
    out[m2] = np.sin(0.5*PI * s2[m2])
    return out


def ctft_meyer(u, v, j=0, k=0, K=4):
    """
    This is the Fourier Transform of the Continuous function f(x,y). F(u,v)
    """
    R = np.hypot(u, v)
    theta = np.arctan2(v, u)

    A = angular_window(theta, K=K, k=k)
    return radial(R * (2 ** j)) * A


def dtft_meyer(U, V, T=3.5/8, j=0, k=0, K=4, physical_scale=False):
    """
    This is the Fourier Transform of the Sampled signal. Sample Speed is T.
    Formula used: DTFT(u,v) =  SUM over n. n in Q: CTFT(u-2*n*PI/T,v-2*n*PI/T) * 1/T*T
    """
    scale = 1.0 / (T * T)

    if T <= NYQUIST_T:
        if physical_scale:
            return scale * ctft_meyer(U/T, V/T, j=j, k=k, K=K)
        return ctft_meyer(U/T, V/T, j=j, k=k, K=K)
    out = np.zeros_like(U, dtype=float)
    if T > NYQUIST_T:
        for mx in range(-1, 2):
            for my in range(-1, 2):
                out += ctft_meyer((U + 2*PI*mx)/T, (V + 2*PI*my)/T, j=j, k=k, K=K)
    if physical_scale:
        return scale * out
    return out


def dft_meyer(N, T=3.5/8, j=0, k=0, K=4, fftshift=True, physical_scale=False):
    """
    This is the DFT of our signal.
    Sample the DTFT using the formula DFT[K] = DTFT[2piK/N]
    """
    if fftshift:
        w = 2*np.pi * np.arange(-N//2, N//2) / N
    else:
        w = 2*np.pi * np.arange(N) / N
    U, V = np.meshgrid(w, w, indexing="xy")

    X = dtft_meyer(U, V, T=T, j=j, k=k, K=K, physical_scale=physical_scale)
    return X


def create_meyer_kernel(domain, N, T, j, k, K, physical_scale=False, S=None):
    """
    Build Meyer kernel (freq or spatial) on an N×N grid.
    If S is given (odd, S<=N) and domain='spatial' (or 'frequency'), return a centered S×S crop.
    """
    X = dft_meyer(N, T, j, k, K, fftshift=True,
                  physical_scale=physical_scale)

    if domain == "frequency":
        # Optional: crop the *centered* frequency patch (DC is already at center)
        return _crop_center(X, S) if S is not None and S < N else X

    elif domain == "spatial":
        # Unshift → IFFT → re-center spatial kernel, then crop.
        h = np.fft.ifft2(np.fft.ifftshift(X))
        h = np.fft.fftshift(h)
        if S is not None and S < N:
            h = _crop_center(h, S)
        return h

    else:
        raise ValueError("domain must be 'frequency' or 'spatial'")


def _crop_center(Z, S):
    """Return the centered SxS crop of a square array Z (S odd, S<=N)."""
    N = Z.shape[0]
    if Z.shape[0] != Z.shape[1]:
        raise ValueError("Z must be square.")
    if S % 2 == 0 or S > N:
        raise ValueError(f"S must be odd and <= N. Got S={S}, N={N}.")
    c = N // 2
    r = S // 2
    return Z[c - r : c + r + 1, c - r : c + r + 1]


def filter_bank(N, T, S, K, j=0):
    """
    Create a filter bank of complex Meyer wavelets at dyadic scale j.
    N must be even, S must be odd. Returns a tensor of shape (K, 1, S, S).
    """
    if N % 2 == 1 or S % 2 == 0:
        raise ValueError(f"[Meyer filter]: N={N} must be even, S={S} must be odd.")

    filters = []
    for k in range(K):
        # Build spatial-domain kernel via Meyer DTFT/DFT pipeline
        ker = create_meyer_kernel(domain="spatial", N=N, T=T, j=j, k=k, K=K, physical_scale=False, S=S)
        ker = torch.tensor(ker, dtype=torch.complex64).unsqueeze(0)
        filters.append(ker)

    return torch.cat(filters, dim=0)  # (K,1,S,S)


def meyer(max_scale, nb_orients, N=64, T=3.5/8, S=7):
    """
    Pyramid of Meyer filter banks across dyadic scales j=0..max_scale-1.
    Returns: list of length max_scale, each item is a (nb_orients,1,S,S) tensor.
    S grows as follows: S -> 2*S + 1 per scale.
    """
    banks = []
    for j in range(max_scale):
        bank = filter_bank(N, T, S, K=nb_orients, j=j)
        banks.append(bank)
        S = 2 * S - 1
    return banks


def crop_center(Z, S):
    """Centered SxS crop. S must be odd."""
    if S % 2 == 0:
        raise ValueError("S must be odd.")
    H, W = Z.shape
    cy, cx = H//2, W//2
    r = S//2
    return Z[cy-r:cy+r+1, cx-r:cx+r+1]


def energy(Z):
    """L2 energy (sum of |Z|^2) as a float."""
    return float(np.sum(np.abs(Z)**2))


def run_meyer_experiment(N=64, T=3.5/8, J=1, S=5, K=4):
    """
    For each j in range (0,1, ..., J), show:
      cols 0, ... ,K-1: per-orientation band-pass at scale j
      col  K : low-pass Φ_J at same scale j
    Rows:
      1) |F|(Ω) frequency response
      2) |f|(x,y) full spatial kernel magnitude (before crop)
      3) |f|(x,y) cropped S×S center patch + kept/loss %
    """

    if S % 2 == 0:
        raise ValueError("S must be odd.")
    plane = np.zeros((N,N))
    w = np.linspace(-np.pi, np.pi, N, endpoint=False)
    extentF = [w.min(), w.max(), w.min(), w.max()]
    for j in range(J+1):
        ncols = K
        fig, axes = plt.subplots(3, ncols, figsize=(3.4*ncols, 9.4), squeeze=False)
        fig.suptitle(f"Meyer filters — scale j={j}, crop S={S}, K={K}",
                     y=0.995, fontsize=14)

        kept_list, loss_list = [], []

        # ---------- Oriented band-pass wedges ----------
        for k in range(K):
            F_jk = create_meyer_kernel(domain="frequency", N=N, T=T, j=j, k=k, K=K)
            f_jk = create_meyer_kernel(domain="spatial",   N=N, T=T, j=j, k=k, K=K)

            plane += np.abs(F_jk)**2  # Add to POU check
            # Row 1: |F|
            axF = axes[0, k]
            imF = axF.imshow(np.abs(F_jk), extent=extentF, origin="lower")
            axF.set_title(f"k={k}  |F|(Ω)")
            axF.set_xlabel("Ωx [rad]"); axF.set_ylabel("Ωy [rad]")
            fig.colorbar(imF, ax=axF, fraction=0.046, pad=0.04)

            # Row 2: |f| full (before crop)
            axSfull = axes[1, k]
            f_full = np.abs(f_jk)
            imSf = axSfull.imshow(f_full, origin="lower")
            axSfull.set_title("|f|(x,y) full"); axSfull.set_xlabel("x"); axSfull.set_ylabel("y")
            fig.colorbar(imSf, ax=axSfull, fraction=0.046, pad=0.04)

            # Row 3: |f| cropped (after crop), + kept/loss %
            axSc = axes[2, k]
            f_crop = crop_center(f_full, S)
            r = S // 2
            imSc = axSc.imshow(f_crop, extent=[-r, r, -r, r], origin="lower")
            axSc.set_xlabel("x"); axSc.set_ylabel("y")
            E_full = energy(f_full)
            E_kept = energy(f_crop)
            frac_kept = (E_kept / E_full) if E_full > 0 else 0.0
            axSc.set_title(f"crop {S}×{S}\nkept={100*frac_kept:.1f}%, loss={100*(1-frac_kept):.1f}%")
            fig.colorbar(imSc, ax=axSc, fraction=0.046, pad=0.04)

            kept_list.append(frac_kept)
            loss_list.append(1.0 - frac_kept)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show(block=False)
        S = 2*S + 1

    fig2, ax2 = plt.subplots(figsize=(5.6, 4.8), facecolor="white")
    im2 = ax2.imshow(plane, origin="lower", interpolation="nearest")
    ax2.set_title(f"Square sum over scales (J={J}) and orientations,{K})")
    ax2.set_xlabel("sample x"); ax2.set_ylabel("sample y")
    fig2.colorbar(im2, ax=ax2, shrink=0.9, pad=0.03).set_label("Sum")

    print(f"[diagnostics] sum min={plane.min():.4g}, max={plane.max():.4g}")
    plt.show(block=False)

    plt.show()


# run_meyer_experiment(N=64, T=3.5/8, J=1, S=5, K=4)

# S = 5 is very good
