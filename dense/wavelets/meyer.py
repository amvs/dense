import numpy as np
import torch
import matplotlib.pyplot as plt
PI = np.pi
NYQUIST_T = 3/8
PERIOD = 2*PI


def smoothing(p):
    # Found on wiki, used in meyer implementations
    t = np.clip(p, 0.0, 1.0)
    return t**4 * (35 - 84*t + 70*t**2 - 20*t**3)


def meyer_radial(rho):
    """
    Classic radial Meyer band: support (2pi/3, 8pi/3), unitless input rho = |ω|.
    """
    out = np.zeros_like(rho)
    m1 = (rho > 2*PI/3) & (rho < 4*PI/3)
    m2 = (rho >= 4*PI/3) & (rho < 8*PI/3)
    out[m1] = np.sin((PI/2) * smoothing(-1 + 3*rho[m1]/(2*PI)))
    out[m2] = np.cos((PI/2) * smoothing(-1 + 3*rho[m2]/(4*PI)))
    return out


def angular_window_pairwise(theta, K, k, eps=1e-12):
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
    R = np.hypot(u, v)
    theta = np.arctan2(v, u)

    A = angular_window_pairwise(theta, K=K, k=k)
    return meyer_radial(R * (2**j)) * A


def dtft_meyer(U, V, T=3/8, j=0, k=0, K=4, physical_scale=False):
    """
    DTFT X_d(w) = (1/T^2) * sum_{mx,my} X_c((wx+2πmx)/T, (wy+2πmy)/T)
    Toggle physical_scale on if you want to multiply by 1/T*T. Disabled to achieve partition of unity.
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


def dft_meyer(N, T=3/8, j=0, k=0, K=4, fftshift=True, physical_scale=False):
    """
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
    X = dft_meyer(N, T, j, k, K, fftshift=True, physical_scale=physical_scale)  # DC at center (freq)  :contentReference[oaicite:0]{index=0}

    if domain == "frequency":
        # Optional: crop the *centered* frequency patch (DC is already at center)
        return _crop_center(X, S) if S is not None and S < N else X

    elif domain == "spatial":
        # Unshift → IFFT → re-center spatial kernel, then crop.
        h = np.fft.ifft2(np.fft.ifftshift(X))                                  # unshift before IFFT  :contentReference[oaicite:1]{index=1}
        h = np.fft.fftshift(h)  # center impulse at (S//2, S//2) to match Morlet’s grid-based kernels
        if S is not None and S < N:
            h = _crop_center(h, S)
        return h

    else:
        raise ValueError("domain must be 'frequency' or 'spatial'")


def low_pass(rho):

    """
    Meyer scaling (low-pass) L(ρ):
      L(ρ)=1                      for ρ ≤ 2π/3
      L(ρ)=cos(π/2 * s(ρ))        for 2π/3 < ρ < 4π/3
      L(ρ)=0                      for ρ ≥ 4π/3
    """

    out = np.zeros_like(rho, float)
    m0 = (rho <= 2 * PI / 3)
    m1 = (rho > 2 * PI / 3) & (rho < 4 * PI / 3)
    out[m0] = 1.0
    out[m1] = np.cos((PI / 2) * smoothing(-1 + 3 * rho[m1] / (2 * PI)))
    return out


def ctft_scaling(u, v, J=0):
    R = np.hypot(u, v)
    return low_pass(R * (2**J))


def dtft_scaling(U, V, T=3/8, J=0, physical_scale=False):
    scale = 1/(T*T)
    if T <= NYQUIST_T:
        if physical_scale:
            return scale * ctft_scaling(U/T, V/T, J)
        return ctft_scaling(U/T, V/T, J)

    out = np.zeros_like(U, dtype=float)
    if T > NYQUIST_T:
        for mx in range(-1, 2):
            for my in range(-1, 2):
                out += ctft_scaling((U + 2*PI*mx)/T, (V + 2*PI*my)/T, J)
    if physical_scale:
        return scale * out
    return out


def dft_scaling(N, T=3/8, J=0, fftshift=True, physical_scale=False):
    if fftshift:
        w = 2*np.pi * np.arange(-N//2, N//2) / N
    else:
        w = 2*np.pi * np.arange(N) / N
    U, V = np.meshgrid(w, w, indexing="xy")

    X = dtft_scaling(U, V, T, J, physical_scale)
    return X


def create_scaling_kernel(domain, N, T, J, physical_scale=False, S=None):
    X = dft_scaling(N, T, J, fftshift=True, physical_scale=physical_scale)  # DC at center
    if domain == "frequency":
        return _crop_center(X, S) if S is not None and S < N else X
    elif domain == "spatial":
        h = np.fft.ifft2(np.fft.ifftshift(X))
        h = np.fft.fftshift(h)
        if S is not None and S < N:
            h = _crop_center(h, S)
        return h
    else:
        raise ValueError("domain must be 'frequency' or 'spatial'")


def plot_meyer(N=64, T=3/8, J_list=(0, 1, 2), k_list=None, K=4, S=None):
    if k_list is None:
        k_list = list(range(K))
    if isinstance(J_list, int): J_list = [J_list]
    if isinstance(k_list, int): k_list = [k_list]

    kernels = []
    if S is None: S = N
    plane = np.zeros((S, S), dtype=float)

    # ----- band-pass kernels & energy sum (spatial, cropped) -----
    for j in J_list:
        row = []
        for k in k_list:
            ker = create_meyer_kernel("spatial", N, T, j, k, K, physical_scale=False, S=S)
            plane += np.abs(ker)**2          # <— use magnitude squared (real)
            row.append(np.real(ker))         # visualize real part
        kernels.append(row)

    # ----- low-pass (spatial, cropped) -----
    j_max = max(J_list)
    lp = create_scaling_kernel("spatial", N, T, j_max, physical_scale=False, S=S)
    plane_lp = np.abs(lp)**2                 # (optional) add to plane if you want DC covered
    plane += plane_lp

    # ----- plotting -----
    nrows, ncols = len(J_list), len(k_list)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4*ncols, 3.2*nrows), squeeze=False, facecolor="white")

    extent = (0, S, 0, S)                    # <— use S since we cropped
    tick_locs = np.linspace(0, S, 5)
    tick_lbls = [f"{int(t)}" for t in tick_locs]

    vmax = max(Z.max() for row in kernels for Z in row)
    vmin = 0.0

    last_im = None
    for i, j in enumerate(J_list):
        for t, k in enumerate(k_list):
            ax = axes[i, t]
            last_im = ax.imshow(kernels[i][t], origin="lower", interpolation="nearest",
                                vmin=vmin, vmax=vmax, extent=extent)
            ax.set_title(f"j={j}, k={k}", fontsize=11)
            ax.set_xlabel("sample x"); ax.set_ylabel("sample y")
            ax.set_xticks(tick_locs); ax.set_yticks(tick_locs)
            ax.set_xticklabels(tick_lbls); ax.set_yticklabels(tick_lbls)

    fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.88, pad=0.02).set_label("Re{kernel}")

    # Low-pass figure
    fig_lp, ax_lp = plt.subplots(figsize=(5.2, 4.6), facecolor="white")
    im_lp = ax_lp.imshow(np.real(lp), origin="lower", interpolation="nearest", extent=extent)
    ax_lp.set_title(f"Low-pass (scaling) kernel at j={j_max}")
    ax_lp.set_xlabel("sample x"); ax_lp.set_ylabel("sample y")
    ax_lp.set_xticks(tick_locs); ax_lp.set_yticks(tick_locs)
    fig_lp.colorbar(im_lp, ax=ax_lp, shrink=0.9, pad=0.03).set_label("Re{scaling}")

    # Energy-sum figure
    fig2, ax2 = plt.subplots(figsize=(5.6, 4.8), facecolor="white")
    im2 = ax2.imshow(plane, origin="lower", interpolation="nearest", extent=extent)
    ax2.set_title("Sum over all (j,k) (|kernel|^2) [+ LP if enabled]")
    ax2.set_xlabel("sample x"); ax2.set_ylabel("sample y")
    ax2.set_xticks(tick_locs); ax2.set_yticks(tick_locs)
    fig2.colorbar(im2, ax=ax2, shrink=0.9, pad=0.03).set_label("Sum")

    print(f"[diagnostics] sum min={plane.min():.4g}, max={plane.max():.4g}")
    plt.show()
    return plane, kernels


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
    # Match morlet.py's interface & checks
    if N % 2 == 1 or S % 2 == 0:
        raise ValueError(f"[Meyer filter]: N={N} must be even, S={S} must be odd.")

    filters = []
    for k in range(K):
        # Build spatial-domain kernel via your Meyer DTFT/DFT pipeline
        ker = create_meyer_kernel(domain="spatial", N=N, T=T, j=j, k=k, K=K, physical_scale=False, S=S)
        ker = torch.tensor(ker, dtype=torch.complex64).unsqueeze(0)  # (1,S,S), like morlet.py
        filters.append(ker)

    return torch.cat(filters, dim=0)  # (K,1,S,S)


def meyer(max_scale, nb_orients, N=64, T=3/8, S=7):
    """
    Pyramid of Meyer filter banks across dyadic scales j=0..max_scale-1.
    Returns: list of length max_scale, each item is a (nb_orients,1,S,S) tensor.
    S grows as in morlet.py: S -> 2*S + 1 per scale.
    """
    banks = []
    for j in range(max_scale):
        bank = filter_bank(N, T, S, K=nb_orients, j=j)
        banks.append(bank)
        S = 2 * S + 1
    return banks


##################################

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


def run_meyer_experiment(N=64, T=3/8, j=0, S=11, K_list=(4, 8), include_lowpass=True):
    """
    For each K in K_list, show:
      cols 0..K-1: per-orientation band-pass at scale j
      col  K     : low-pass Φ_J at same scale j (if include_lowpass)
    Rows:
      1) |F|(Ω) frequency response
      2) |f|(x,y) full spatial kernel magnitude (before crop)
      3) |f|(x,y) cropped S×S center patch + kept/loss %
    """
    if S % 2 == 0:
        raise ValueError("S must be odd.")

    # frequency axis extent (DC centered; your frequency kernels are returned with DC at center)
    w = np.linspace(-np.pi, np.pi, N, endpoint=False)
    extentF = [w.min(), w.max(), w.min(), w.max()]

    for K in K_list:
        ncols = K + (1 if include_lowpass else 0)
        fig, axes = plt.subplots(3, ncols, figsize=(3.4*ncols, 9.4), squeeze=False)
        fig.suptitle(f"Meyer filters — scale j={j}, crop S={S}, K={K}" + (" (+LP)" if include_lowpass else ""),
                     y=0.995, fontsize=14)

        kept_list, loss_list = [], []

        # ---------- Oriented band-pass wedges ----------
        for k in range(K):
            # Frequency response (DC at center by construction)
            F_jk = create_meyer_kernel(domain="frequency", N=N, T=T, j=j, k=k, K=K)
            # Spatial kernel (your create_meyer_kernel already centers the impulse)
            f_jk = create_meyer_kernel(domain="spatial",   N=N, T=T, j=j, k=k, K=K)

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

        # ---------- Low-pass Φ_J in the last column (optional) ----------
        if include_lowpass:
            col = K
            F_lp = create_scaling_kernel(domain="frequency", N=N, T=T, J=j)  # DC at center
            f_lp = create_scaling_kernel(domain="spatial",   N=N, T=T, J=j)  # may be origin-centered in your code

            # If your create_scaling_kernel doesn't fftshift the spatial kernel,
            # display the shifted version so it is centered:
            f_lp_full = np.abs(np.fft.fftshift(f_lp))

            # Row 1: |Φ_J|(Ω)
            axF = axes[0, col]
            imF = axF.imshow(np.abs(F_lp), extent=extentF, origin="lower")
            axF.set_title(r"LP  $|\Phi_J|$"); axF.set_xlabel("Ωx"); axF.set_ylabel("Ωy")
            fig.colorbar(imF, ax=axF, fraction=0.046, pad=0.04)

            # Row 2: |φ_J|(x,y) full (before crop)
            axSfull = axes[1, col]
            imSf = axSfull.imshow(f_lp_full, origin="lower")
            axSfull.set_title(r"LP  $|\phi_J|$ full"); axSfull.set_xlabel("x"); axSfull.set_ylabel("y")
            fig.colorbar(imSf, ax=axSfull, fraction=0.046, pad=0.04)

            # Row 3: cropped
            axSc = axes[2, col]
            f_lp_crop = crop_center(f_lp_full, S)
            r = S // 2
            imSc = axSc.imshow(f_lp_crop, extent=[-r, r, -r, r], origin="lower")
            E_full_lp = energy(f_lp_full); E_kept_lp = energy(f_lp_crop)
            frac_kept_lp = (E_kept_lp / E_full_lp) if E_full_lp > 0 else 0.0
            axSc.set_title(f"LP crop {S}×{S}\nkept={100*frac_kept_lp:.1f}%, loss={100*(1-frac_kept_lp):.1f}%")
            axSc.set_xlabel("x"); axSc.set_ylabel("y")
            fig.colorbar(imSc, ax=axSc, fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

# run_meyer_experiment()
##################################

# plot_meyer(S=63)
