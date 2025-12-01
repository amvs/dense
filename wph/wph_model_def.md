# Wavelet phase harmonic (WPH) Model

## Correlation Model

WPH model is a correlation model, designed for textures.
We use correlations for textures because we model textures as realizations of a stationary, ergodic stochastic process.
Given a representation $Rx( \gamma)$, where $\gamma$ is parameters that define representation like scale and rotation and $x$ is the input signal, the correlation (aka covariance) is given by:
$$ Cx ( \gamma, \gamma', \tau )  = \int_{\Omega} Rx (\gamma, u) Rx(\gamma', u-\tau)^* du$$
where $\Omega$ is the domain. $u$ is the spatial location and $\tau$ is the spatial shift.

If $\Omega$ is a discrete domain (e.g. an image),

$$ Cx ( \gamma, \gamma', \tau )  = \frac{1}{|\Omega|} \sum_{u \in \Omega} Rx (\gamma, u) Rx(\gamma', u-\tau)^* du$$

This is a general framework that works for many representations of a signal.
The specifics of the model depend on the representation.

## WPH Representation

Choose a mother wavlet $\psi: \R^2 \mapsto \mathbb{C}$.
Choose maximum scale $J$ and number of angles $L$; this gives scales $j \in \{ 0, 1, \ldots, J-1 \}$ and rotations $r_{\theta}$ over angles $\theta = l \pi / L$ for $l \in \{ 0 , 1, \ldots, L \}$.
Let $\Delta = \{ 0, 1,\ldots, J-1 \} \times \frac{\pi}{L} \{ 0, \ldots, L-1 \}$ be the index set for the wavelets.
Then we have wavelet with scale $j$ and angle $\theta$ given by
$$ \psi_{k, \theta} (u) = 2^{-2j} \psi( 2^{-2j} r_{\theta} u), u \in \R^2$$
The wavelet coefficients for input signal $x$ are given by:
$$ x \star \psi_{j, \theta}(u) = \sum_{v \in \Omega} x(u-v) \psi_{j, \theta}(v), \quad u \in \Omega, (j,\theta) \in \Delta $$

Then the WPH representation is given by:
$$ R^{WPH} = [ x \star \psi_{j, \theta}(u) ]^k - \mu_{\gamma}, \quad \gamma \in \Delta \times k$$
where $\mu_{\gamma}$ is the spatial average of $[ x \star \psi_{\gamma}(u) ]$ and $[z]^k$ is the $k$-th phase harmonic of complex number $z$ defined by:
$$[z]^k = |z| e^{i k \phi(z)}$$

## Alpha Representation

However, in practice, we don't actually use the phase harmonics, because they capture the same information as phase-shifted ReLU coefficients.

Define phase-shifted ReLU by:
$$ \rho_{\alpha}(z) = \rho(\mathcal{R}(e^{i \alpha}z))$$
where $\rho$ is ReLU and $\mathcal{R}$ denotes taking the real part.

Then the Alpha model representation is given by:
$$R^{Alpha} x (\gamma, u) = \rho_{\alpha} (x \star \psi_{j, \theta}(u)) - \mu_{\gamma}, \quad \gamma  = (j, \theta, \alpha) $$

### Implementation Note: Filter Phase-Shifting

In practice, we phase-shift the filters before convolution rather than phase-shifting the coefficients after. These are equivalent due to linearity of convolution. Let $z = x \star \psi_{j,\theta}(u)$. Then:
$$x \star (e^{i\alpha}\psi_{j,\theta})(u) = e^{i\alpha}(x \star \psi_{j,\theta}(u)) $$

Therefore:
$$\rho(\mathcal{R}(x \star (e^{i\alpha}\psi_{j,\theta})(u))) = \rho(\mathcal{R}(e^{i\alpha} z)) = \rho_\alpha(z)$$

This equivalence allows us to precompute phase-shifted filters $e^{i\alpha}\psi_{j,\theta}$ and use them directly in a CNN architecture.

## Architecture

The WPH model is implemented as a three-layer neural network architecture.

**Layer 1: Convolutional Layer**

A convolutional layer with $|\Gamma|$ filters, where $\Gamma = \{ 0, 1, \ldots, J-1 \} \times \frac{\pi}{L} \{ 0, \ldots, L-1 \} \times \{ \alpha_1, \ldots, \alpha_K \}$. Each filter corresponds to a phase-shifted wavelet $e^{i\alpha}\psi_{j,\theta}$. The real part is taken as input to the next layer.

**Layer 2: ReLU Activation and Mean Centering**

Apply ReLU activation followed by subtraction of the spatial mean $\mu_\gamma$ for each channel $\gamma$, producing $R^{Alpha} x (\gamma, u)$.

**Layer 3: Correlation Layer**

Compute pairwise correlations between channels at spatial shifts $\tau \in \mathcal{T}$:
$$C^{WPH} x (\gamma, \gamma', \tau) = \frac{1}{|\Omega|} \sum_{u \in \Omega} R^{Alpha} x (\gamma, u) \cdot R^{Alpha} x (\gamma', u-\tau)$$

Only compute correlations for $(\gamma, \gamma') \in \mathcal{P}$, where $\mathcal{P}$ typically includes:
- All auto-correlations: $(\gamma, \gamma)$ for $\gamma \in \Gamma$
- Selected cross-correlations: $(\gamma, \gamma')$ with $|j - j'| \leq \Delta_j$ for some maximum scale difference

The output is $\Phi^{WPH}(x) = \{ C^{WPH} x (\gamma, \gamma', \tau) : (\gamma, \gamma') \in \mathcal{P}, \tau \in \mathcal{T} \}$.

### Implementation Notes:
- The convolutional filters can be initialized with wavelet functions $e^{i\alpha}\psi_{j,\theta}$ (for WPH model) or randomly initialized (for trainable networks)
- Filters are treated as learnable parameters that can be updated via gradient descent
- The sets $\mathcal{P}$ and $\mathcal{T}$ are hyperparameters that control the dimensionality of the output

## References

This notation is taken from the paper in which the WPH model is proposed and applied to texture generation: 
```
@inproceedings{
brochard2022generalized,
title={Generalized rectifier wavelet covariance models for texture synthesis},
author={Antoine Brochard and Sixin Zhang},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=ziRLU3Y2PN_}
}
```