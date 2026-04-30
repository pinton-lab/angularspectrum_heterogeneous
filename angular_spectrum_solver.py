"""
Angular Spectrum Solver — refactored Python/JAX port of angular_spectrum_solver.m

Implements:
  - Strang split-step propagation (2nd-order splitting accuracy)
  - TVD slope limiting (generalized minmod) for the nonlinear flux
  - Adaptive transverse-wavenumber filtering in the angular spectrum operator
  - Intensity-loss tracking due to attenuation
  - Wendland C2 smooth boundary profiles
  - Frequency-weighted boundary damping (quasi-PML)
  - Super-absorbing boundary condition (Engquist-Majda directional decomposition)

Original MATLAB code by Gianmarco Pinton (2017-2023).
Python port and refactoring 2024-2026.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from dataclasses import dataclass, field
from typing import Optional, Callable
import time as _time

from tof_extraction import (
    extract_tof_envelope as _extract_tof,
    extract_tof_matched_filter_parabolic as _extract_tof_mf,
)


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------
@dataclass
class SolverParams:
    """All solver parameters with defaults matching the MATLAB refactored code."""
    dX: float = 0.0
    dY: float = 0.0
    dT: float = 0.0
    c0: float = 1500.0
    rho0: float = 1000.0
    beta: float = 3.5
    alpha0: float = 0.5          # dB/MHz^pow/cm  (negative → water)
    attenPow: float = 1.0
    f0: float = 5e6
    propDist: float = 0.08
    boundaryFactor: float = 0.2
    useSplitStep: bool = True
    useAdaptiveFiltering: bool = True   # k-space cosine taper
    useTVD: bool = True                  # TVD slope limiter (independent of k-filtering)
    freqFilterThreshold: float = 0.05
    adaptiveFilterStrength: float = 0.7
    stabilityThreshold: float = 0.2
    stabilityRecoveryFactor: float = 0.15
    dZmin: float = 1e-3
    # --- boundary improvements ---
    useBoundaryLayer: bool = True
    boundaryProfile: str = 'quadratic'   # 'quadratic' | 'wendland'
    useFreqWeightedBoundary: bool = False
    useSuperAbsorbing: bool = False
    superAbsorbingStrength: float = 0.8  # 0–1, fraction of incoming wave removed
    # --- flux scheme ---
    fluxScheme: str = 'rusanov'  # 'rusanov' | 'kt'  (Kurganov-Tadmor)
    # --- obliquity correction on attenuation/dispersion filter ---
    useObliquityCorrection: bool = True  # scale alpha, alphaStar by k/k_z per mode
    # --- beam-averaged obliquity correction on the nonlinear operator ---
    # Scales the Burgers coefficient N by the power-weighted <k/k_z> of the
    # current plane-wave spectrum, so that the effective nonlinear path length
    # matches the beam's mean obliquity rather than the axial step dz. Adds one
    # FFT per march step when enabled.
    useNonlinearityObliquity: bool = False
    # --- phase screens for heterogeneous propagation ---
    phaseScreens: object = None  # list of (z_position, screen_array) tuples
    # --- distributed source injection (bowl transducer) ---
    sourcePlanes: object = None  # list of (z_position, field_slice) from make_bowl_source_planes


# ---------------------------------------------------------------------------
# Absorbing boundary profiles
# ---------------------------------------------------------------------------
def ablvec(N: int, n: int) -> np.ndarray:
    """Original quadratic boundary profile: C0 continuous."""
    vec = np.zeros(N)
    for nn in range(n):
        x = (n - nn - 1) / n
        vec[nn] = x ** 2
    for nn in range(N - n, N):
        x = (nn - (N - n - 1)) / n
        vec[nn] = x ** 2
    return 1.0 - vec


def ablvec_wendland(N: int, n: int) -> np.ndarray:
    """Wendland C2 boundary profile: 1 - 10s^3 + 15s^4 - 6s^5.

    The value, first derivative, and second derivative are all zero at
    the domain edge (s=1) and smoothly transition to 1 in the interior
    (s=0).  This greatly reduces spurious reflections compared to the
    quadratic profile whose second derivative is discontinuous at the
    taper onset.
    """
    vec = np.ones(N)
    for nn in range(n):
        s = (n - nn - 1) / n          # 1 at edge, 0 at interior
        vec[nn] = 1.0 - 10*s**3 + 15*s**4 - 6*s**5
    for nn in range(N - n, N):
        s = (nn - (N - n - 1)) / n
        vec[nn] = 1.0 - 10*s**3 + 15*s**4 - 6*s**5
    return np.clip(vec, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Precalculate absorbing boundary layer (spatial × temporal)
# ---------------------------------------------------------------------------
def precalculate_abl(nX, nY, nT, boundary_factor=0.2, split_step=False,
                     profile='quadratic'):
    print(f'Precalculating absorbing boundary layer ({profile})...')
    _vec_fn = ablvec_wendland if profile == 'wendland' else ablvec

    x_bw = max(round(nX * boundary_factor), 1)
    y_bw = max(round(nY * boundary_factor), 1)
    t_bw = max(round(nT * boundary_factor), 1)

    abl_tmp = np.outer(_vec_fn(nX, x_bw), _vec_fn(nY, y_bw))
    abl_vec = _vec_fn(nT, t_bw)
    abl = (abl_tmp[:, :, np.newaxis] * abl_vec[np.newaxis, np.newaxis, :]).astype(np.float32)

    abl_half = np.sqrt(abl) if split_step else None
    print('done.')
    return abl, abl_half


def precalculate_identity_abl(nX, nY, nT, split_step=False):
    """Return a neutral boundary mask for studies that should exclude boundary damping."""
    abl = np.ones((nX, nY, nT), dtype=np.float32)
    abl_half = abl.copy() if split_step else None
    return abl, abl_half


# ---------------------------------------------------------------------------
# Frequency-weighted boundary damping  (improvement 3 — quasi-PML)
# ---------------------------------------------------------------------------
def precalculate_abl_freq(nX, nY, nT, dT, c0, f0, boundary_factor=0.2,
                          profile='quadratic'):
    """Precompute a frequency-dependent spatial damping mask.

    At low temporal frequencies the boundary is fewer wavelengths wide,
    so the damping must be stronger to prevent reflection.  The mask is
    stored in the rfft frequency layout: shape (nX, nY, nT//2+1).

    For each frequency bin f_m the 2D spatial taper is raised to the
    power  s_m = f_ref / max(f_m, f_min)  so that:
      * f_m = f_ref  →  normal damping (exponent 1)
      * f_m < f_ref  →  stronger damping (exponent > 1)
      * f_m > f_ref  →  weaker damping   (exponent < 1)
    """
    print('Precalculating frequency-weighted boundary mask...')
    _vec_fn = ablvec_wendland if profile == 'wendland' else ablvec

    x_bw = max(round(nX * boundary_factor), 1)
    y_bw = max(round(nY * boundary_factor), 1)
    abl_xy = np.outer(_vec_fn(nX, x_bw), _vec_fn(nY, y_bw))  # (nX, nY)

    n_freq = nT // 2 + 1
    freqs = np.fft.rfftfreq(nT, dT)         # (n_freq,)
    f_ref = f0
    f_min = freqs[1] if len(freqs) > 1 else 1.0   # avoid div-by-zero

    abl_freq = np.zeros((nX, nY, n_freq), dtype=np.float32)
    for m in range(n_freq):
        fm = max(freqs[m], f_min)
        exponent = f_ref / fm
        # Clamp exponent to avoid extreme values
        exponent = np.clip(exponent, 0.1, 10.0)
        abl_freq[:, :, m] = abl_xy ** exponent

    print('done.')
    return abl_freq


# ---------------------------------------------------------------------------
# Super-absorbing boundary (improvement 4 — Engquist-Majda directional)
# ---------------------------------------------------------------------------
@jit
def _super_absorbing_step(field, bdy_weight_x, bdy_weight_y, dX, dY, dT,
                          c0, strength):
    """Remove the incoming wave component at each spatial boundary.

    Uses a first-order Engquist-Majda decomposition in the temporal
    frequency domain.  At each frequency omega the incoming pressure at
    the left x-boundary is

        P_in = 0.5 * (P  +  (c0 / (i omega)) * dP/dx )

    and at the right boundary

        P_in = 0.5 * (P  -  (c0 / (i omega)) * dP/dx )

    The incoming component is then subtracted (scaled by *strength*) in
    the boundary region defined by bdy_weight_x / bdy_weight_y.

    Parameters
    ----------
    field        : (nX, nY, nT) real field
    bdy_weight_x : (nX,)  spatial weight: 0 in interior, >0 in boundary
    bdy_weight_y : (nY,)  spatial weight
    dX, dY, dT   : grid spacings
    c0           : sound speed
    strength     : 0-1, fraction of incoming wave removed
    """
    nX, nY, nT_full = field.shape

    # Transform to temporal frequency domain (rfft along axis 2)
    F = jnp.fft.rfft(field, axis=2)                  # (nX, nY, n_freq)
    n_freq = F.shape[2]

    # Frequency vector (angular)
    freqs = jnp.fft.rfftfreq(nT_full, dT)
    omega = 2.0 * jnp.pi * freqs                     # (n_freq,)

    # Coefficient  c0 / (i omega) = -i c0 / omega
    # At DC (omega=0) and very low frequencies the decomposition is
    # ill-conditioned, so we zero the coefficient there.
    omega_min = 2.0 * jnp.pi * freqs[1] if n_freq > 1 else 1.0
    valid = jnp.abs(omega) > 0.5 * omega_min
    coeff = jnp.where(valid, -1j * c0 / jnp.where(valid, omega, 1.0), 0.0 + 0j)
    coeff = coeff[jnp.newaxis, jnp.newaxis, :]        # (1, 1, n_freq)

    # --- x-boundaries ---
    # Forward spatial gradient dF/dx  (central differences, one-sided at edges)
    dFdx = jnp.zeros_like(F)
    dFdx = dFdx.at[1:-1, :, :].set((F[2:, :, :] - F[:-2, :, :]) / (2 * dX))
    dFdx = dFdx.at[0, :, :].set((F[1, :, :] - F[0, :, :]) / dX)
    dFdx = dFdx.at[-1, :, :].set((F[-1, :, :] - F[-2, :, :]) / dX)

    # Incoming at left  (positive x direction = into domain)
    P_in_left = 0.5 * (F + coeff * dFdx)
    # Incoming at right (negative x direction = into domain)
    P_in_right = 0.5 * (F - coeff * dFdx)

    # Build weight for left boundary (bdy_weight > 0 near edges)
    # Left half gets left correction, right half gets right correction
    wx = bdy_weight_x                                 # (nX,)
    wx_left  = wx * (jnp.arange(nX) < nX // 2)
    wx_right = wx * (jnp.arange(nX) >= nX // 2)
    wx_left  = wx_left[:, jnp.newaxis, jnp.newaxis]   # (nX,1,1)
    wx_right = wx_right[:, jnp.newaxis, jnp.newaxis]

    F = F - strength * wx_left  * P_in_left
    F = F - strength * wx_right * P_in_right

    # --- y-boundaries ---
    dFdy = jnp.zeros_like(F)
    dFdy = dFdy.at[:, 1:-1, :].set((F[:, 2:, :] - F[:, :-2, :]) / (2 * dY))
    dFdy = dFdy.at[:, 0, :].set((F[:, 1, :] - F[:, 0, :]) / dY)
    dFdy = dFdy.at[:, -1, :].set((F[:, -1, :] - F[:, -2, :]) / dY)

    P_in_bottom = 0.5 * (F + coeff * dFdy)
    P_in_top    = 0.5 * (F - coeff * dFdy)

    wy = bdy_weight_y                                 # (nY,)
    wy_bottom = wy * (jnp.arange(nY) < nY // 2)
    wy_top    = wy * (jnp.arange(nY) >= nY // 2)
    wy_bottom = wy_bottom[jnp.newaxis, :, jnp.newaxis]
    wy_top    = wy_top[jnp.newaxis, :, jnp.newaxis]

    F = F - strength * wy_bottom * P_in_bottom
    F = F - strength * wy_top    * P_in_top

    # Transform back
    return jnp.fft.irfft(F, n=nT_full, axis=2)


def _make_boundary_weights(N, n_bdy):
    """Smooth weight that is 1 at the edge and 0 in the interior."""
    w = np.zeros(N, dtype=np.float32)
    for nn in range(n_bdy):
        s = (n_bdy - nn - 1) / n_bdy  # 1 at edge → 0 at interior boundary
        w[nn] = s
    for nn in range(N - n_bdy, N):
        s = (nn - (N - n_bdy - 1)) / n_bdy
        w[nn] = s
    return w


# ---------------------------------------------------------------------------
# Phase screen model for heterogeneous propagation
# ---------------------------------------------------------------------------
def generate_phase_screen(nX, nY, dX, dY, c0, f0,
                          correlation_length, speed_std,
                          thickness=None, seed=None):
    """Generate a random phase screen for modelling tissue heterogeneity.

    The screen represents a thin layer of tissue with spatially-varying
    sound speed.  The speed fluctuations are drawn from a Gaussian random
    field with a specified correlation length and standard deviation.

    Parameters
    ----------
    nX, nY           : int     — grid dimensions
    dX, dY           : float   — grid spacing (m)
    c0               : float   — background sound speed (m/s)
    f0               : float   — center frequency (Hz) — sets k0
    correlation_length : float — spatial correlation length (m)
    speed_std        : float   — standard deviation of sound-speed
                                 fluctuations (m/s)
    thickness        : float   — effective layer thickness (m);
                                 default = correlation_length
    seed             : int     — random seed for reproducibility

    Returns
    -------
    phase_shift : ndarray (nX, nY) — phase shift in radians at each
                  grid point.  Applied to the field as
                  p *= exp(i * phase_shift).
    c_map       : ndarray (nX, nY) — the sound-speed map (m/s).
    """
    if thickness is None:
        thickness = correlation_length
    rng = np.random.default_rng(seed)

    # Spatial frequency grid
    kx = np.fft.fftfreq(nX, dX) * 2 * np.pi
    ky = np.fft.fftfreq(nY, dY) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2

    # Gaussian power spectrum with correlation length L:
    #   S(k) ∝ exp(-k² L² / 4)
    L = correlation_length
    spectrum = np.exp(-K2 * L**2 / 4.0)

    # Generate complex white noise and filter
    noise = rng.standard_normal((nX, nY)) + 1j * rng.standard_normal((nX, nY))
    filtered = np.fft.ifft2(np.fft.fft2(noise) * np.sqrt(spectrum))
    delta_c = np.real(filtered)

    # Normalize to desired standard deviation
    delta_c = delta_c / (np.std(delta_c) + 1e-30) * speed_std

    # Sound speed map
    c_map = c0 + delta_c

    # Phase shift: Δφ = ω * thickness * (1/c(x,y) - 1/c0)
    omega = 2 * np.pi * f0
    phase_shift = omega * thickness * (1.0 / c_map - 1.0 / c0)

    return phase_shift.astype(np.float32), c_map.astype(np.float32)


@jit
def _apply_phase_screen(field, phase_screen, f0_bin, amplitude_screen=None):
    """Apply a phase+amplitude screen in the temporal frequency domain.

    The phase screen shifts each frequency component by phase * f/f0.
    The amplitude screen attenuates each frequency component by
    amp^(f/f0) — higher frequencies see more attenuation through bone.

    Parameters
    ----------
    field : (nX, nY, nT) real-valued pressure field
    phase_screen : (nX, nY) phase shift at f0 in radians
    f0_bin : float — the rfft bin index corresponding to f0
    amplitude_screen : (nX, nY) transmission factor at f0 (0 to 1), optional
    """
    nT = field.shape[2]
    F = jnp.fft.rfft(field, axis=2)          # (nX, nY, n_freq)
    n_freq = F.shape[2]

    # Frequency bin indices; f_bin = k * df, f0 = f0_bin * df
    # Scale: f / f0 = k / f0_bin
    freq_idx = jnp.arange(n_freq, dtype=jnp.float32)
    f_scale = freq_idx / jnp.maximum(f0_bin, 1.0)

    # Phase modulation: exp(-i * phase_screen(x,y) * f/f0)
    # Negative sign: phase = omega*dz*(1/c - 1/c0) is the extra spatial
    # wavenumber*distance.  In the temporal rfft domain the correction is
    # exp(-j * delta_phi) because spatial propagation uses exp(-j*kz*z).
    ps = phase_screen[:, :, jnp.newaxis]     # (nX, nY, 1)
    fs = f_scale[jnp.newaxis, jnp.newaxis, :]  # (1, 1, n_freq)
    F = F * jnp.exp(-1j * ps * fs)

    # Amplitude modulation: amp^(f/f0) — frequency-dependent attenuation
    if amplitude_screen is not None:
        amp = amplitude_screen[:, :, jnp.newaxis]  # (nX, nY, 1)
        # amp^(f/f0): at f0 get the nominal attenuation, higher f more loss
        F = F * jnp.power(jnp.clip(amp, 1e-6, 1.0), fs)

    return jnp.fft.irfft(F, n=nT, axis=2)


# ---------------------------------------------------------------------------
# Precalculate modified angular spectrum (with optional adaptive filtering)
# ---------------------------------------------------------------------------
def precalculate_mas(nX, nY, nT, dX, dY, dZ, dT, c0,
                     split_step=False,
                     adaptive_filtering=False,
                     filter_threshold=0.05,
                     filter_strength=0.7):
    print('Precalculating modified angular spectrum...')

    kt = np.linspace(0, nT - 1, nT) / (nT - 1) / dT * 2 * np.pi / c0
    kt -= np.mean(kt)
    kx = np.linspace(0, nX - 1, nX) / (nX - 1) / dX * 2 * np.pi
    kx -= np.mean(kx)
    ky = np.linspace(0, nY - 1, nY) / (nY - 1) / dY * 2 * np.pi
    ky -= np.mean(ky)

    # Transverse wavenumber grid (precomputed once)
    kk = kx[:, None] ** 2 + ky[None, :] ** 2  # (nX, nY)

    # Adaptive filtering setup — kmax is the actual max spatial freq in the centered grid
    kmax = min(np.pi / dX, np.pi / dY)
    lambda_char = c0 / np.mean(np.abs(kt[kt != 0])) if np.any(kt != 0) else c0 / (1.0 / dT)
    norm_step = dZ / lambda_char

    if adaptive_filtering:
        cutoff = filter_threshold * (1 + filter_strength * norm_step * 5)
        k_trans_max = min(kmax * (1 - cutoff), kmax * 0.95)
        k_trans_start = k_trans_max * 0.8
        print(f'  adaptive filtering: cutoff={cutoff:.4f}, norm_step={norm_step:.4f}')
    else:
        k_trans_max = kmax * 0.95
        k_trans_start = k_trans_max * 0.9

    k_trans = np.sqrt(kk)
    filt_mask = np.ones_like(k_trans)
    trans = (k_trans > k_trans_start) & (k_trans <= k_trans_max)
    if np.any(trans):
        norm_pos = (k_trans[trans] - k_trans_start) / (k_trans_max - k_trans_start)
        filt_mask[trans] = 0.5 * (1 + np.cos(np.pi * norm_pos))
    filt_mask[k_trans > k_trans_max] = 0.0

    HH = np.zeros((nX, nY, nT), dtype=np.complex128)
    HH_half = np.zeros((nX, nY, nT), dtype=np.complex128) if split_step else None

    for m in range(nT):
        k = kt[m]
        H2 = np.exp(dZ * (1j * k - np.sqrt(kk - k ** 2 + 0j)))
        H1 = np.exp(dZ * (1j * k - 1j * np.sqrt(k ** 2 - kk + 0j)))
        H = np.where(kk < k ** 2, H1, H2)
        H *= filt_mask
        HH[:, :, m] = H

        if split_step:
            H2h = np.exp(dZ / 2 * (1j * k - np.sqrt(kk - k ** 2 + 0j)))
            H1h = np.exp(dZ / 2 * (1j * k - 1j * np.sqrt(k ** 2 - kk + 0j)))
            Hh = np.where(kk < k ** 2, H1h, H2h)
            Hh *= filt_mask
            HH_half[:, :, m] = Hh

    # Zero negative frequencies & DC bin, then double positive frequencies.
    # +1 so the DC bin (index nT/2 for even nT) is included in the zeroed half.
    HH[:, :, :nT // 2 + 1] = 0
    HH *= 2
    if split_step:
        HH_half[:, :, :nT // 2 + 1] = 0
        HH_half *= 2

    print('done.')
    return HH, HH_half


# ---------------------------------------------------------------------------
# Attenuation / dispersion
# ---------------------------------------------------------------------------
def _build_oblique_atten_filter(alpha, alphaStar, nX, nY, nT, dX, dY, dZ, c0,
                                dT, scale=1.0, obliquity=True,
                                clip_negative=True):
    """Construct the obliquity-corrected attenuation/dispersion filter.

    Per plane-wave mode (k_x, k_y, omega), the physical path length over one
    axial step dZ is dZ/cos(theta) = dZ * k/k_z, where k = omega/c_0 and
    k_z = sqrt(k^2 - k_perp^2). Applying alpha and alphaStar as pure
    omega-only filters underestimates attenuation for oblique components;
    multiplying the argument by k/k_z restores the correct path length.

    Returned array has shape (nX, nY, nT), complex, in fftshifted-centered
    (k_x, k_y) order and natural temporal-frequency order (matches the HH
    propagator convention; _attenuation_step slices to the rfft support).

    For evanescent modes (k_perp^2 > k^2), no obliquity correction is
    applied (the diffraction propagator already decays these); for the
    DC/zero-frequency bin the filter is unity.
    """
    # Transverse wavenumber grid in the same centered convention as HH
    kx = np.linspace(0, nX - 1, nX) / (nX - 1) / dX * 2 * np.pi
    kx -= np.mean(kx)
    ky = np.linspace(0, nY - 1, nY) / (nY - 1) / dY * 2 * np.pi
    ky -= np.mean(ky)
    kk = kx[:, None] ** 2 + ky[None, :] ** 2  # (nX, nY)

    f = np.arange(nT) / (nT * dT)
    omega = 2 * np.pi * f
    k = omega / c0  # |k| per temporal-frequency bin

    afilt3d = np.ones((nX, nY, nT), dtype=np.complex128)
    eps = 1e-6
    for m in range(nT):
        km = k[m]
        if km == 0:
            continue  # DC: no attenuation, filter stays unity
        base = -(alpha[m] + 1j * alphaStar[m]) * dZ * scale
        if obliquity:
            kz_sq = km ** 2 - kk
            prop = kz_sq > 0
            kz_safe = np.where(prop, np.maximum(np.sqrt(np.maximum(kz_sq, 0.0)),
                                                np.abs(km) * eps), 1.0)
            obl = np.where(prop, np.abs(km) / kz_safe, 0.0)
            factor = np.exp(base * obl)
            # Evanescent modes: no extra attenuation beyond what HH does
            factor = np.where(prop, factor, 1.0)
        else:
            factor = np.broadcast_to(np.exp(base), kk.shape)
        afilt3d[:, :, m] = factor

    if clip_negative:
        afilt3d = np.where(np.real(afilt3d) < 0, 0, afilt3d)
    return afilt3d


def precalculate_ad(alpha0, nX, nY, nT, dX, dY, dZ, dT, c0, f0,
                    split_step=False, obliquity=True):
    print('Precalculating attenuation filter (water, f^2, '
          f'{"obliquity-corrected" if obliquity else "omega-only"})...')
    f = np.arange(nT) / (nT * dT)
    conv = alpha0 / 1e12 * 1e2 / (20 * np.log10(np.e))
    alpha = conv * f ** 2

    alphaStar0 = (conv / (2 * np.pi) ** 2) * np.tan(np.pi) * \
                 ((2 * np.pi * f) ** 1 - (2 * np.pi * f0) ** 1)
    alphaStar = 2 * np.pi * alphaStar0 * f
    alphaStar[0] = 0

    afilt3d = _build_oblique_atten_filter(alpha, alphaStar, nX, nY, nT,
                                          dX, dY, dZ, c0, dT,
                                          scale=1.0, obliquity=obliquity)

    afilt3d_half = None
    if split_step:
        afilt3d_half = _build_oblique_atten_filter(alpha, alphaStar,
                                                   nX, nY, nT,
                                                   dX, dY, dZ, c0, dT,
                                                   scale=0.5,
                                                   obliquity=obliquity)

    print('done.')
    return afilt3d, afilt3d_half


# ---------------------------------------------------------------------------
# Attenuation / dispersion — general power law
# ---------------------------------------------------------------------------
def precalculate_ad_pow2(alpha0, nX, nY, nT, dX, dY, dZ, dT, c0, f0, pw,
                         split_step=False, obliquity=True):
    print(f'Precalculating attenuation filter (f^{pw} power law, '
          f'{"obliquity-corrected" if obliquity else "omega-only"})...')
    f = np.arange(nT) / (nT * dT)
    f_safe = f.copy()
    f_safe[0] = f_safe[1] / 2  # avoid log(0)

    conv = alpha0 / (1e6 ** pw) * 1e2 / (20 * np.log10(np.e))
    alpha = conv * f ** pw

    if pw % 2 == 1:
        alphaStar0 = (-2 * conv / ((2 * np.pi) ** pw) / np.pi) * \
                     (np.log(2 * np.pi * f_safe) - np.log(2 * np.pi * f0))
    else:
        alphaStar0 = (conv / (2 * np.pi) ** pw) * np.tan(np.pi * pw / 2) * \
                     ((2 * np.pi * f) ** (pw - 1) - (2 * np.pi * f0) ** (pw - 1))

    alphaStar = 2 * np.pi * alphaStar0 * f
    alphaStar[0] = 0

    afilt3d = _build_oblique_atten_filter(alpha, alphaStar, nX, nY, nT,
                                          dX, dY, dZ, c0, dT,
                                          scale=1.0, obliquity=obliquity,
                                          clip_negative=False)

    afilt3d_half = None
    if split_step:
        afilt3d_half = _build_oblique_atten_filter(alpha, alphaStar,
                                                   nX, nY, nT,
                                                   dX, dY, dZ, c0, dT,
                                                   scale=0.5,
                                                   obliquity=obliquity,
                                                   clip_negative=False)

    dispersion = 1.0 / ((1.0 / c0) + (alphaStar / (2 * np.pi * f_safe)))
    print('done.')
    return afilt3d, afilt3d_half, f, alpha, dispersion


# ---------------------------------------------------------------------------
# Beam-averaged obliquity for the nonlinear operator
# ---------------------------------------------------------------------------
def precalculate_obliquity_map(nX, nY, nT, dX, dY, dT, c0):
    """Per-mode 1/cos(theta) = k/k_z on the rfft temporal grid.

    Returns a float32 array of shape (nX, nY, nT//2+1) in fftshifted-centered
    (k_x, k_y) order, zero for evanescent modes and for the DC bin.
    """
    kx = np.linspace(0, nX - 1, nX) / (nX - 1) / dX * 2 * np.pi
    kx -= np.mean(kx)
    ky = np.linspace(0, nY - 1, nY) / (nY - 1) / dY * 2 * np.pi
    ky -= np.mean(ky)
    kk = kx[:, None] ** 2 + ky[None, :] ** 2  # (nX, nY)

    n_freq = nT // 2 + 1
    f = np.arange(n_freq) / (nT * dT)
    k_tf = 2 * np.pi * f / c0  # (n_freq,)

    obl_map = np.zeros((nX, nY, n_freq), dtype=np.float32)
    eps = 1e-6
    for m in range(n_freq):
        km = k_tf[m]
        if km == 0:
            continue
        kz_sq = km ** 2 - kk
        prop = kz_sq > 0
        kz_safe = np.where(prop,
                           np.maximum(np.sqrt(np.maximum(kz_sq, 0.0)),
                                      np.abs(km) * eps),
                           1.0)
        obl_map[:, :, m] = np.where(prop, np.abs(km) / kz_safe, 0.0).astype(np.float32)
    return obl_map


@jit
def _beam_obliquity_scalar(field, obl_map):
    """Power-weighted <k/k_z> over the propagating plane-wave content.

    Returns a scalar in [1, inf); equals 1 for an axially-propagating beam and
    grows as rim rays take larger angles from z.
    """
    F = jnp.fft.rfft(field, axis=2)               # (nX, nY, nT//2+1)
    F = jnp.fft.fftshift(jnp.fft.fft2(F, axes=(0, 1)), axes=(0, 1))
    P = (F.real ** 2 + F.imag ** 2).astype(obl_map.dtype)
    prop_mask = (obl_map > 0).astype(obl_map.dtype)
    num = jnp.sum(P * obl_map)
    denom = jnp.sum(P * prop_mask) + 1e-30
    return num / denom


# ---------------------------------------------------------------------------
# JIT-compiled march steps
# ---------------------------------------------------------------------------
@jit
def _angular_spectrum_step(field, HH, abl):
    """Full-step angular spectrum propagation + spatial/temporal boundary layer."""
    field = jnp.real(jnp.fft.ifftn(
        jnp.fft.ifftshift(jnp.fft.fftshift(jnp.fft.fftn(field)) * HH)
    ))
    return field * abl


@jit
def _freq_weighted_boundary_step(field, abl_freq):
    """Apply frequency-dependent spatial damping in the rfft domain."""
    nT = field.shape[2]
    F = jnp.fft.rfft(field, axis=2)
    F = F * abl_freq
    return jnp.fft.irfft(F, n=nT, axis=2)


@jit
def _attenuation_step(field, afilt3d):
    """Obliquity-corrected attenuation/dispersion.

    afilt3d has shape (nX, nY, nT//2+1), complex, in fftshifted-centered
    (k_x, k_y) order (same convention as the HH propagator). For each
    propagating plane-wave mode the filter carries a k/k_z factor that
    applies alpha and alphaStar over the correct per-mode path length
    dZ/cos(theta); evanescent modes see an identity filter since the
    diffraction step already handles their decay.
    """
    nT = field.shape[2]
    F = jnp.fft.rfft(field, axis=2)               # (nX, nY, nT//2+1) — (x, y, omega)
    F = jnp.fft.fft2(F, axes=(0, 1))              # (k_x, k_y, omega), natural order
    F = jnp.fft.fftshift(F, axes=(0, 1))          # centered (k_x, k_y)
    F = F * afilt3d
    F = jnp.fft.ifftshift(F, axes=(0, 1))
    F = jnp.fft.ifft2(F, axes=(0, 1))
    return jnp.fft.irfft(F, n=nT, axis=2).real


@jit
def _rusanov_flux_standard(field, N, dZ, dT):
    """Standard Rusanov flux (no TVD limiting)."""
    lambdahalf = jnp.maximum(jnp.abs(field[:, :, :-1]),
                             jnp.abs(field[:, :, 1:]))
    fluxhalf = -(field[:, :, :-1] ** 2 + field[:, :, 1:] ** 2) / 2 - \
               lambdahalf * (field[:, :, 1:] - field[:, :, :-1])
    flux_diff = fluxhalf[:, :, 1:] - fluxhalf[:, :, :-1]
    return field.at[:, :, 1:-1].add(-N * dZ / dT * flux_diff)


@jit
def _rusanov_flux_tvd(field, N, dZ, dT, beta_tvd):
    """Rusanov flux with TVD generalized minmod slope limiting."""
    uL = field[:, :, :-1]
    uR = field[:, :, 1:]
    a = jnp.maximum(jnp.abs(uL), jnp.abs(uR))
    fluxhalf = 0.5 * (-0.5 * (uL ** 2 + uR ** 2)) - 0.5 * a * (uR - uL)

    # Flux differences at interior points
    flux_diff = fluxhalf[:, :, 1:] - fluxhalf[:, :, :-1]

    # TVD generalized minmod limiter
    delta_minus = field[:, :, 1:-1] - field[:, :, :-2]
    delta_plus = field[:, :, 2:] - field[:, :, 1:-1]

    eps = 1e-30
    r = delta_plus / (delta_minus + eps * jnp.sign(delta_minus + eps))
    r_inv = delta_minus / (delta_plus + eps * jnp.sign(delta_plus + eps))

    phi_plus = jnp.maximum(0.0, jnp.minimum(jnp.minimum(beta_tvd * r, 1.0),
                                              jnp.minimum(r, beta_tvd)))
    phi_minus = jnp.maximum(0.0, jnp.minimum(jnp.minimum(beta_tvd * r_inv, 1.0),
                                               jnp.minimum(r_inv, beta_tvd)))

    limited_flux = 0.5 * (phi_plus + phi_minus) * flux_diff
    return field.at[:, :, 1:-1].add(-N * dZ / dT * limited_flux)


@jit
def _minmod(a, b):
    """Two-argument minmod limiter."""
    same_sign = a * b > 0
    return jnp.where(same_sign, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)


@jit
def _minmod3(a, b, c):
    """Three-argument minmod: returns the value with smallest magnitude
    if all three have the same sign, else zero."""
    same_sign = (a * b > 0) & (b * c > 0)
    s = jnp.sign(a)
    m = jnp.minimum(jnp.minimum(jnp.abs(a), jnp.abs(b)), jnp.abs(c))
    return jnp.where(same_sign, s * m, 0.0)


@jit
def _mc_limiter(delta_minus, delta_plus):
    """Monotonized central (MC) limiter — less diffusive than minmod.

    Returns minmod(2*delta_minus, (delta_minus+delta_plus)/2, 2*delta_plus).
    Uses the centered slope when safe; falls back to one-sided slopes
    near discontinuities.  TVD for any convex combination of minmod
    and centered differences.
    """
    return _minmod3(2 * delta_minus,
                    0.5 * (delta_minus + delta_plus),
                    2 * delta_plus)


@jit
def _kt_rhs(field, N, dT):
    """Semidiscrete KT right-hand side with MUSCL-MC reconstruction.

    Uses the MC (monotonized central) limiter for sharper shock
    resolution while retaining the TVD property.
    """
    # Limited slopes in each cell along the time axis.
    delta_minus = field[:, :, 1:-1] - field[:, :, :-2]
    delta_plus = field[:, :, 2:] - field[:, :, 1:-1]
    sigma_mid = _mc_limiter(delta_minus, delta_plus)
    sigma = jnp.pad(sigma_mid, ((0, 0), (0, 0), (1, 1)))

    # MUSCL reconstruction at interfaces j+1/2.
    u_minus = field[:, :, :-1] + 0.5 * sigma[:, :, :-1]
    u_plus = field[:, :, 1:] - 0.5 * sigma[:, :, 1:]

    # Local one-sided wave speeds for retarded-time Burgers: f'(u) = -u.
    a_plus = jnp.maximum(jnp.maximum(-u_minus, -u_plus), 0.0)
    a_minus = jnp.minimum(jnp.minimum(-u_minus, -u_plus), 0.0)

    # Retarded-time Burgers flux f(u) = -u^2 / 2.
    f_minus = -0.5 * u_minus ** 2
    f_plus = -0.5 * u_plus ** 2

    denom = a_plus - a_minus
    safe_denom = jnp.where(jnp.abs(denom) < 1e-14, 1.0, denom)
    kt_flux = (
        a_plus * f_minus
        - a_minus * f_plus
        - a_plus * a_minus * (u_plus - u_minus)
    ) / safe_denom

    # Fall back to Lax-Friedrichs when the local speeds collapse.
    lf_flux = 0.5 * (f_minus + f_plus) - 0.5 * jnp.maximum(jnp.abs(u_minus), jnp.abs(u_plus)) * (u_plus - u_minus)
    fluxhalf = jnp.where(jnp.abs(denom) < 1e-14, lf_flux, kt_flux)

    flux_diff = fluxhalf[:, :, 1:] - fluxhalf[:, :, :-1]
    rhs = jnp.zeros_like(field)
    return rhs.at[:, :, 1:-1].set(-N / dT * flux_diff)


@jit
def _kt_rhs_minmod(field, N, dT):
    """KT right-hand side with the original minmod limiter."""
    delta_minus = field[:, :, 1:-1] - field[:, :, :-2]
    delta_plus = field[:, :, 2:] - field[:, :, 1:-1]
    sigma_mid = _minmod(delta_minus, delta_plus)
    sigma = jnp.pad(sigma_mid, ((0, 0), (0, 0), (1, 1)))

    u_minus = field[:, :, :-1] + 0.5 * sigma[:, :, :-1]
    u_plus = field[:, :, 1:] - 0.5 * sigma[:, :, 1:]

    a_plus = jnp.maximum(jnp.maximum(-u_minus, -u_plus), 0.0)
    a_minus = jnp.minimum(jnp.minimum(-u_minus, -u_plus), 0.0)

    f_minus = -0.5 * u_minus ** 2
    f_plus = -0.5 * u_plus ** 2

    denom = a_plus - a_minus
    safe_denom = jnp.where(jnp.abs(denom) < 1e-14, 1.0, denom)
    kt_flux = (
        a_plus * f_minus
        - a_minus * f_plus
        - a_plus * a_minus * (u_plus - u_minus)
    ) / safe_denom

    lf_flux = 0.5 * (f_minus + f_plus) - 0.5 * jnp.maximum(jnp.abs(u_minus), jnp.abs(u_plus)) * (u_plus - u_minus)
    fluxhalf = jnp.where(jnp.abs(denom) < 1e-14, lf_flux, kt_flux)

    flux_diff = fluxhalf[:, :, 1:] - fluxhalf[:, :, :-1]
    rhs = jnp.zeros_like(field)
    return rhs.at[:, :, 1:-1].set(-N / dT * flux_diff)


@jit
def _kt_flux(field, N, dZ, dT):
    """Second-order KT nonlinear step with SSP-RK2 integration in z."""
    rhs0 = _kt_rhs(field, N, dT)
    stage1 = field + dZ * rhs0
    rhs1 = _kt_rhs(stage1, N, dT)
    return 0.5 * field + 0.5 * (stage1 + dZ * rhs1)


def _kt_flux_adaptive(field, N, dZ, dT, cfl_target=0.5):
    """KT nonlinear step with adaptive sub-cycling for CFL stability.

    If the CFL number N*|u_max|*dZ/dT exceeds cfl_target, the step is
    split into sub-steps that each satisfy the CFL condition.  Each
    sub-step uses SSP-RK2 for second-order accuracy.
    """
    u_max = float(jnp.max(jnp.abs(field)))
    cfl = N * u_max * dZ / dT
    if cfl <= cfl_target or u_max < 1e-30:
        return _kt_flux(field, N, dZ, dT)

    # Sub-cycle: split dZ into n_sub steps
    n_sub = int(jnp.ceil(cfl / cfl_target))
    dz_sub = dZ / n_sub
    for _ in range(n_sub):
        field = _kt_flux(field, N, dz_sub, dT)
    return field


@jit
def _kt_flux_minmod(field, N, dZ, dT):
    """KT with minmod limiter (for comparison)."""
    rhs0 = _kt_rhs_minmod(field, N, dT)
    stage1 = field + dZ * rhs0
    rhs1 = _kt_rhs_minmod(stage1, N, dT)
    return 0.5 * field + 0.5 * (stage1 + dZ * rhs1)


# ---------------------------------------------------------------------------
# Composite march steps
# ---------------------------------------------------------------------------
@jit
def march_step_sequential(field, HH, abl, afilt3d, N, dZ, dT):
    """Original Lie (sequential) splitting — 1st-order."""
    field = _angular_spectrum_step(field, HH, abl)
    field = _rusanov_flux_standard(field, N, dZ, dT)
    field = _attenuation_step(field, afilt3d)
    return field


@jit
def march_step_split_standard(field, HH_half, abl_half, afilt3d_half, N, dZ, dT):
    """Strang split-step with standard Rusanov flux (no TVD)."""
    field = _attenuation_step(field, afilt3d_half)
    field = _angular_spectrum_step(field, HH_half, abl_half)
    field = _rusanov_flux_standard(field, N, dZ, dT)
    field = _angular_spectrum_step(field, HH_half, abl_half)
    field = _attenuation_step(field, afilt3d_half)
    return field


@partial(jit, static_argnums=())
def march_step_split_tvd(field, HH_half, abl_half, afilt3d_half, N, dZ, dT, beta_tvd):
    """Strang split-step with TVD-limited Rusanov flux."""
    field = _attenuation_step(field, afilt3d_half)
    field = _angular_spectrum_step(field, HH_half, abl_half)
    field = _rusanov_flux_tvd(field, N, dZ, dT, beta_tvd)
    field = _angular_spectrum_step(field, HH_half, abl_half)
    field = _attenuation_step(field, afilt3d_half)
    return field


@jit
def march_step_split_kt(field, HH_half, abl_half, afilt3d_half, N, dZ, dT):
    """Strang split-step with second-order KT nonlinear flux."""
    field = _attenuation_step(field, afilt3d_half)
    field = _angular_spectrum_step(field, HH_half, abl_half)
    field = _kt_flux(field, N, dZ, dT)
    field = _angular_spectrum_step(field, HH_half, abl_half)
    field = _attenuation_step(field, afilt3d_half)
    return field


# --- variants with beam-averaged obliquity correction on the nonlinear step ---
# The Burgers coefficient N is scaled by <k/k_z>, the power-weighted mean
# obliquity of the current plane-wave spectrum entering the nonlinear step.
# This corrects the effective path length dz -> dz/<cos(theta)> so that a
# strongly-focused beam whose rim rays travel at angles theta from z
# accumulates nonlinear steepening over the correct physical distance.

@jit
def march_step_sequential_obl(field, HH, abl, afilt3d, obl_map, N, dZ, dT):
    field = _angular_spectrum_step(field, HH, abl)
    N_eff = N * _beam_obliquity_scalar(field, obl_map)
    field = _rusanov_flux_standard(field, N_eff, dZ, dT)
    field = _attenuation_step(field, afilt3d)
    return field


@jit
def march_step_split_standard_obl(field, HH_half, abl_half, afilt3d_half,
                                  obl_map, N, dZ, dT):
    field = _attenuation_step(field, afilt3d_half)
    field = _angular_spectrum_step(field, HH_half, abl_half)
    N_eff = N * _beam_obliquity_scalar(field, obl_map)
    field = _rusanov_flux_standard(field, N_eff, dZ, dT)
    field = _angular_spectrum_step(field, HH_half, abl_half)
    field = _attenuation_step(field, afilt3d_half)
    return field


@partial(jit, static_argnums=())
def march_step_split_tvd_obl(field, HH_half, abl_half, afilt3d_half,
                             obl_map, N, dZ, dT, beta_tvd):
    field = _attenuation_step(field, afilt3d_half)
    field = _angular_spectrum_step(field, HH_half, abl_half)
    N_eff = N * _beam_obliquity_scalar(field, obl_map)
    field = _rusanov_flux_tvd(field, N_eff, dZ, dT, beta_tvd)
    field = _angular_spectrum_step(field, HH_half, abl_half)
    field = _attenuation_step(field, afilt3d_half)
    return field


@jit
def march_step_split_kt_obl(field, HH_half, abl_half, afilt3d_half,
                            obl_map, N, dZ, dT):
    field = _attenuation_step(field, afilt3d_half)
    field = _angular_spectrum_step(field, HH_half, abl_half)
    N_eff = N * _beam_obliquity_scalar(field, obl_map)
    field = _kt_flux(field, N_eff, dZ, dT)
    field = _angular_spectrum_step(field, HH_half, abl_half)
    field = _attenuation_step(field, afilt3d_half)
    return field


# ---------------------------------------------------------------------------
# Source generation helpers
# ---------------------------------------------------------------------------
def make_bowl_source(xaxis, yaxis, taxis, f0, c0, p0,
                     radius, roc, ncycles=3, dur=2,
                     inner_radius=0.0, focus=None,
                     element_delays=None, n_elements=1):
    """Create the initial field for a spherical bowl (focused) transducer.

    Supports solid bowls, annular apertures (inner_radius > 0), and
    multi-element annular arrays with per-element steering delays.

    Parameters
    ----------
    xaxis, yaxis : 1-D arrays
        Lateral grid coordinates (m).
    taxis : 1-D array
        Time axis (s), centred near zero.
    f0 : float
        Centre frequency (Hz).
    c0 : float
        Sound speed (m/s).
    p0 : float
        Source pressure amplitude (Pa).
    radius : float
        Outer aperture radius of the bowl (m).
    roc : float
        Radius of curvature of the bowl (m).
    ncycles : float
        Number of cycles in the Gaussian-windowed tone burst.
    dur : int
        Super-Gaussian exponent for the temporal envelope.
    inner_radius : float
        Inner radius of an annular aperture (m).  Set to 0 for a solid bowl.
    focus : float or None
        Electronic focal depth (m) from the bowl apex.  If None, defaults
        to ``roc`` (geometric focus).  For TIPS with ROC=80 mm focused at
        50 mm, set roc=80e-3 and focus=50e-3.
    element_delays : array-like or None
        Additional per-element time delays (s) for electronic steering.
        Length must equal n_elements.  These are added on top of the
        focusing delays.
    n_elements : int
        Number of annular elements (concentric rings of equal radial width).

    Returns
    -------
    field : ndarray, shape (nX, nY, nT)
        Initial pressure on the source plane with focusing encoded as
        per-pixel time delays.
    """
    if focus is None:
        focus = roc
    nX = len(xaxis)
    nY = len(yaxis)
    omega0 = 2 * np.pi * f0
    R = roc

    XX, YY = np.meshgrid(xaxis, yaxis, indexing='ij')
    r2 = XX**2 + YY**2
    r = np.sqrt(r2)

    # Annular aperture
    aperture = ((r2 <= radius**2) & (r2 >= inner_radius**2)).astype(np.float32)

    # Bowl surface depth: z_bowl(r) = R - sqrt(R² - r²), apex at z=0
    r2_clipped = np.minimum(r2, R**2 - 1e-20)
    z_bowl = R - np.sqrt(R**2 - r2_clipped)

    # Distance from each bowl surface point to the focal point at (0, 0, focus)
    dist_to_focus = np.sqrt(r2 + (focus - z_bowl)**2)

    # Delays: invert so that outermost elements fire first (converging)
    delays = np.zeros_like(r)
    delays[aperture > 0] = dist_to_focus[aperture > 0] / c0

    # Average per element (matching FDTD approach)
    if n_elements > 1:
        dr = (radius - inner_radius) / n_elements
        for ie in range(n_elements):
            r_lo = inner_radius + ie * dr
            r_hi = inner_radius + (ie + 1) * dr
            mask = (r >= r_lo) & (r < r_hi) & (aperture > 0)
            if mask.any():
                delays[mask] = delays[mask].mean()
                if element_delays is not None:
                    delays[mask] += np.asarray(element_delays)[ie]

    # Invert: outermost fires first
    delays[aperture > 0] = delays[aperture > 0].max() - delays[aperture > 0]

    # Build space-time field with per-pixel delays
    t_grid = taxis[np.newaxis, np.newaxis, :]
    d_grid = delays[:, :, np.newaxis]
    envelope = np.exp(-(1.05 * (t_grid - d_grid) * omega0
                        / (ncycles * np.pi))**(2 * dur))
    field = aperture[:, :, np.newaxis] * envelope \
            * np.sin(omega0 * (t_grid - d_grid)) * p0

    n_active = int(np.sum(aperture > 0))
    active_delays = delays[aperture > 0]
    delay_spread = float(np.max(active_delays) - np.min(active_delays))
    print(f'  Bowl source: outer_r={radius*1e3:.1f} mm, '
          f'inner_r={inner_radius*1e3:.1f} mm, '
          f'ROC={R*1e3:.1f} mm, focus={focus*1e3:.1f} mm, '
          f'{n_elements} elements, '
          f'{n_active} active grid points, '
          f'delay spread: {delay_spread*1e6:.2f} us')

    return field.astype(np.float32)


def make_bowl_source_planes(xaxis, yaxis, taxis, f0, c0, p0,
                            radius, roc, dZ, ncycles=3, dur=2,
                            inner_radius=0.0, focus=None,
                            element_delays=None, n_elements=1):
    """Create source planes for plane-by-plane injection of a bowl transducer.

    Instead of projecting the entire bowl onto a single plane, this function
    slices the spherical cap along the propagation axis and returns one
    source field per z-slice.  The solver injects each slice as it
    propagates through the corresponding depth.

    Parameters
    ----------
    xaxis, yaxis : 1-D arrays
        Lateral grid coordinates (m).
    taxis : 1-D array
        Time axis (s), centred near zero.
    f0, c0, p0 : float
        Centre frequency (Hz), sound speed (m/s), pressure amplitude (Pa).
    radius : float
        Outer aperture radius (m).
    roc : float
        Radius of curvature of the bowl surface (m).
    dZ : float
        Axial step size (m) — determines the thickness of each slice.
    ncycles, dur : float, int
        Temporal pulse parameters.
    inner_radius : float
        Inner radius for annular aperture (m).
    focus : float or None
        Electronic focal depth (m) from bowl apex.  Defaults to ``roc``.
    element_delays : array-like or None
        Per-element steering delays (s).
    n_elements : int
        Number of annular elements.

    Returns
    -------
    source_planes : list of (z_position, field_slice)
        Each entry is a tuple of (z, ndarray(nX, nY, nT)) giving the
        source contribution to add at that propagation depth.  z = 0
        is the bowl apex (shallowest point).
    bowl_depth : float
        Total depth of the bowl (m).
    """
    if focus is None:
        focus = roc
    nX = len(xaxis)
    nY = len(yaxis)
    omega0 = 2 * np.pi * f0
    R = roc

    XX, YY = np.meshgrid(xaxis, yaxis, indexing='ij')
    r2 = XX**2 + YY**2
    r = np.sqrt(r2)

    # Annular aperture mask
    aperture = ((r2 <= radius**2) & (r2 >= inner_radius**2))

    # Bowl surface depth: z(r) = R - sqrt(R² - r²)
    r2_clipped = np.minimum(r2, R**2 - 1e-20)
    bowl_z = R - np.sqrt(R**2 - r2_clipped)
    bowl_z[~aperture] = 0.0
    bowl_depth = float(np.max(bowl_z[aperture]))

    # Distance from each bowl surface point to the electronic focal point
    dist_to_focus = np.sqrt(r2 + (focus - bowl_z)**2)
    delays_full = np.zeros_like(r)
    delays_full[aperture] = dist_to_focus[aperture] / c0

    # Average per element
    if n_elements > 1:
        dr = (radius - inner_radius) / n_elements
        for ie in range(n_elements):
            r_lo = inner_radius + ie * dr
            r_hi = inner_radius + (ie + 1) * dr
            mask = (r >= r_lo) & (r < r_hi) & aperture
            if mask.any():
                delays_full[mask] = delays_full[mask].mean()
                if element_delays is not None:
                    delays_full[mask] += np.asarray(element_delays)[ie]

    # Invert so outermost fires first
    delays_full[aperture] = delays_full[aperture].max() - delays_full[aperture]

    # Slice the bowl into z-layers
    n_slices = max(1, int(np.ceil(bowl_depth / dZ)))
    z_edges = np.linspace(0, bowl_depth, n_slices + 1)

    source_planes = []
    total_active = 0

    for i in range(n_slices):
        z_lo = z_edges[i]
        z_hi = z_edges[i + 1]
        z_mid = 0.5 * (z_lo + z_hi)

        slice_mask = aperture & (bowl_z >= z_lo) & (bowl_z < z_hi)
        if i == n_slices - 1:
            slice_mask = aperture & (bowl_z >= z_lo) & (bowl_z <= z_hi)

        n_px = int(np.sum(slice_mask))
        if n_px == 0:
            continue

        # Use the same delays but account for the propagation distance
        # already covered by the solver when it reaches this z-slice.
        # The pulse from this slice has already propagated z_bowl from apex,
        # so advance its timing by z_bowl/c0.
        delays = delays_full.copy()
        delays[slice_mask] -= bowl_z[slice_mask] / c0
        delays[~slice_mask] = 0.0

        # Build the field for this slice
        slice_ap = slice_mask.astype(np.float32)
        t_grid = taxis[np.newaxis, np.newaxis, :]
        d_grid = delays[:, :, np.newaxis]
        envelope = np.exp(-(1.05 * (t_grid - d_grid) * omega0
                            / (ncycles * np.pi))**(2 * dur))
        slice_field = slice_ap[:, :, np.newaxis] * envelope \
                      * np.sin(omega0 * (t_grid - d_grid)) * p0

        source_planes.append((z_mid, slice_field.astype(np.float32)))
        total_active += n_px

    print(f'  Bowl source (plane-by-plane): outer_r={radius*1e3:.1f} mm, '
          f'inner_r={inner_radius*1e3:.1f} mm, '
          f'ROC={R*1e3:.1f} mm, focus={focus*1e3:.1f} mm, '
          f'bowl_depth={bowl_depth*1e3:.2f} mm, '
          f'{n_slices} z-slices, {total_active} active grid points')

    return source_planes, bowl_depth


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
def angular_spectrum_solve(
        initial_field: np.ndarray,
        params: SolverParams,
        verbose: bool = True,
        per_step_callback: Optional[Callable[[int, float, "jnp.ndarray"], None]] = None,
        tof_ref_trace: Optional[np.ndarray] = None,
        tof_t_ref_peak_s: Optional[float] = None,
        tof_env_ratio: Optional[float] = None,
        taxis: Optional[np.ndarray] = None):
    """
    Propagate a 3-D acoustic field using the modified angular spectrum method.

    Parameters
    ----------
    initial_field : ndarray, shape (nX, nY, nT)
        Initial pressure field at the source plane.
    params : SolverParams
        Physical and numerical parameters.
    verbose : bool
        Print progress.
    per_step_callback : callable or None
        If provided, called as ``per_step_callback(cc, z_cumulative, field)``
        after each march step. ``cc`` is the step index, ``z_cumulative``
        is the total propagated distance (m) up to and including the
        current step, and ``field`` is the JAX array at the end of the
        step. No-op when ``None`` (default); introduces a single ``is
        None`` check per step when unused.
    tof_ref_trace : ndarray (nT,) or None
        Reference pulse used by the matched-filter-parabolic estimator
        (``tof_extraction.extract_tof_matched_filter_parabolic``). When
        this is provided, TOF extraction is enabled and this is the
        preferred method (sub-sample accuracy). Requires
        ``tof_t_ref_peak_s``. Takes precedence over ``tof_env_ratio``
        if both are given.
    tof_t_ref_peak_s : float or None
        Time (s) at which ``tof_ref_trace`` itself peaks (envelope-
        argmax time of the reference). Required when ``tof_ref_trace``
        is provided.
    tof_env_ratio : float or None
        Envelope-based fallback: if provided (and ``tof_ref_trace`` is
        not), the solver computes per-(x, y) TOF at each z-step using
        ``tof_extraction.extract_tof_envelope`` with this threshold
        fraction (``1.0`` picks the envelope-peak time).
    taxis : ndarray (nT,) or None
        Time axis used by the TOF extractor. If ``None``, defaults to
        ``np.arange(nT) * params.dT``. Only consulted when TOF
        extraction is enabled (``tof_ref_trace`` or ``tof_env_ratio``
        is not ``None``).

    TOF extraction is enabled when either ``tof_ref_trace`` or
    ``tof_env_ratio`` is not ``None``; otherwise the returned tuple
    has the original 7-element layout.

    Returns
    -------
    field : ndarray  (nX, nY, nT) — final pressure field
    pnp   : ndarray  (nX, nY, nZ) — peak negative pressure
    ppp   : ndarray  (nX, nY, nZ) — peak positive pressure
    pI    : ndarray  (nX, nY, nZ) — time-integrated intensity
    pIloss: ndarray  (nX, nY, nZ) — intensity loss due to attenuation
    zaxis : ndarray  (nZ,)        — propagation coordinates
    pax   : ndarray  (nT, nZ)     — axial pressure trace
    tof   : ndarray  (nX, nY, nZ) — time-of-flight at each step, only
                                    returned when ``tof_env_ratio`` is
                                    not ``None``.
    """
    jax.config.update("jax_enable_x64", True)

    nX, nY, nT = initial_field.shape
    c0 = params.c0
    rho0 = params.rho0
    N = params.beta / (2 * c0 ** 3 * rho0)
    alpha0_eff = params.alpha0
    pw = params.attenPow

    # --- initial step-size estimate ---
    max_amp = np.max(np.abs(initial_field))
    dZ = min(params.stabilityRecoveryFactor * params.dT / (max_amp * N + 1e-30),
             params.dZmin) if max_amp > 0 else params.dZmin
    dZ = max(dZ, params.dZmin)

    # --- precalculate operators ---
    bdy_profile = params.boundaryProfile

    def _build_operators(dZ):
        HH, HH_half = precalculate_mas(
            nX, nY, nT, params.dX, params.dY, dZ, params.dT, c0,
            split_step=params.useSplitStep,
            adaptive_filtering=params.useAdaptiveFiltering,
            filter_threshold=params.freqFilterThreshold,
            filter_strength=params.adaptiveFilterStrength)
        if params.useBoundaryLayer:
            abl, abl_half = precalculate_abl(
                nX, nY, nT, params.boundaryFactor,
                split_step=params.useSplitStep, profile=bdy_profile)
        else:
            abl, abl_half = precalculate_identity_abl(
                nX, nY, nT, split_step=params.useSplitStep)

        if alpha0_eff < 0:  # water
            a0w = 2.17e-3
            afilt3d, afilt3d_half = precalculate_ad(
                a0w, nX, nY, nT, params.dX, params.dY, dZ, params.dT,
                c0, params.f0,
                split_step=params.useSplitStep,
                obliquity=params.useObliquityCorrection)
        else:
            afilt3d, afilt3d_half, _, _, _ = precalculate_ad_pow2(
                alpha0_eff, nX, nY, nT, params.dX, params.dY, dZ, params.dT,
                c0, params.f0, pw,
                split_step=params.useSplitStep,
                obliquity=params.useObliquityCorrection)

        # Convert to rfft-compatible shapes (positive frequencies only)
        def _to_rfft(arr):
            if arr is None:
                return None
            n_freq = nT // 2 + 1
            return arr[:, :, :n_freq]

        return (HH, HH_half, abl, abl_half,
                _to_rfft(afilt3d), _to_rfft(afilt3d_half))

    HH, HH_half, abl_np, abl_half_np, afilt3d_np, afilt3d_half_np = _build_operators(dZ)

    def _to_jax(HH, HH_half, abl_np, abl_half_np, afilt3d_np, afilt3d_half_np):
        ops = dict(
            HH=jnp.array(HH, dtype=jnp.complex64),
            abl=jnp.array(abl_np, dtype=jnp.float32),
            afilt3d=jnp.array(afilt3d_np, dtype=jnp.complex64),
        )
        if params.useSplitStep:
            ops['HH_half'] = jnp.array(HH_half, dtype=jnp.complex64)
            ops['abl_half'] = jnp.array(abl_half_np, dtype=jnp.float32)
            ops['afilt3d_half'] = jnp.array(afilt3d_half_np, dtype=jnp.complex64)

        # Frequency-weighted boundary (quasi-PML)
        if params.useFreqWeightedBoundary:
            abl_freq_np = precalculate_abl_freq(
                nX, nY, nT, params.dT, c0, params.f0,
                boundary_factor=params.boundaryFactor, profile=bdy_profile)
            ops['abl_freq'] = jnp.array(abl_freq_np, dtype=jnp.float32)

        # Super-absorbing boundary weights
        if params.useSuperAbsorbing:
            x_bw = max(round(nX * params.boundaryFactor), 1)
            y_bw = max(round(nY * params.boundaryFactor), 1)
            ops['bdy_weight_x'] = jnp.array(
                _make_boundary_weights(nX, x_bw), dtype=jnp.float32)
            ops['bdy_weight_y'] = jnp.array(
                _make_boundary_weights(nY, y_bw), dtype=jnp.float32)

        return ops

    ops = _to_jax(HH, HH_half, abl_np, abl_half_np, afilt3d_np, afilt3d_half_np)

    # Beam-averaged obliquity map for nonlinearity correction (optional)
    if params.useNonlinearityObliquity:
        obl_map_np = precalculate_obliquity_map(
            nX, nY, nT, params.dX, params.dY, params.dT, c0)
        ops['obl_map'] = jnp.array(obl_map_np, dtype=jnp.float32)

    # Compute f0 bin index for phase screen application
    df = 1.0 / (nT * params.dT)  # frequency resolution
    f0_bin = params.f0 / df       # rfft bin index of center frequency

    # --- TOF setup ---
    # Enabled when either a reference pulse (matched-filter) or
    # tof_env_ratio (envelope) is provided. Matched-filter takes
    # precedence when both are set.
    _tof_use_mf = tof_ref_trace is not None
    _tof_enabled = _tof_use_mf or (tof_env_ratio is not None)
    if _tof_enabled:
        _tof_t = (np.asarray(taxis) if taxis is not None
                  else np.arange(nT, dtype=np.float64) * params.dT)
        if _tof_use_mf:
            if tof_t_ref_peak_s is None:
                raise ValueError(
                    'tof_t_ref_peak_s is required when tof_ref_trace is provided')
            _tof_ref = np.asarray(tof_ref_trace)
            _tof_t_ref_peak = float(tof_t_ref_peak_s)

    # --- allocate output ---
    max_steps = int(np.ceil(params.propDist / dZ)) + 100
    pnp = np.zeros((nX, nY, max_steps), dtype=np.float32)
    ppp = np.zeros((nX, nY, max_steps), dtype=np.float32)
    pI = np.zeros((nX, nY, max_steps), dtype=np.float32)
    pIloss = np.zeros((nX, nY, max_steps), dtype=np.float32)
    pax = np.zeros((nT, max_steps), dtype=np.float32)
    tof = np.zeros((nX, nY, max_steps), dtype=np.float32) if _tof_enabled else None

    field = jnp.array(initial_field, dtype=jnp.float32)
    zvec = []
    cc = 0
    t0 = _time.time()

    while sum(zvec) < params.propDist - 1e-15:
        remaining = params.propDist - sum(zvec)
        dZ_step = min(dZ, remaining)

        # --- current state and stability check ---
        field_np = np.array(field)
        max_field = np.max(np.abs(field_np))
        if N * dZ_step / params.dT * max_field > params.stabilityThreshold:
            if verbose:
                print('Stability criterion violated — reducing step size')
            dZ = params.stabilityRecoveryFactor * params.dT / (max_field * N)
            dZ = max(dZ, params.dZmin)
            cc = 0
            zvec = []
            field = jnp.array(initial_field, dtype=jnp.float32)
            ops_tuple = _build_operators(dZ)
            HH, HH_half, abl_np, abl_half_np, afilt3d_np, afilt3d_half_np = ops_tuple
            ops = _to_jax(*ops_tuple)
            max_steps = int(np.ceil(params.propDist / dZ)) + 100
            pnp = np.zeros((nX, nY, max_steps), dtype=np.float32)
            ppp = np.zeros((nX, nY, max_steps), dtype=np.float32)
            pI = np.zeros((nX, nY, max_steps), dtype=np.float32)
            pIloss = np.zeros((nX, nY, max_steps), dtype=np.float32)
            pax = np.zeros((nT, max_steps), dtype=np.float32)
            if _tof_enabled:
                tof = np.zeros((nX, nY, max_steps), dtype=np.float32)
            continue

        if verbose:
            print(f'z = {sum(zvec) + dZ_step:.6f} m  (step {cc})')

        # --- propagate one step ---
        I_before = np.sum(field_np ** 2, axis=2)

        step_ops = ops
        if abs(dZ_step - dZ) > 1e-15:
            step_ops_tuple = _build_operators(dZ_step)
            step_ops = _to_jax(*step_ops_tuple)

        use_obl_nl = params.useNonlinearityObliquity
        obl_map_op = ops.get('obl_map') if use_obl_nl else None
        if params.useSplitStep:
            if params.fluxScheme == 'kt':
                if use_obl_nl:
                    field = march_step_split_kt_obl(
                        field, step_ops['HH_half'], step_ops['abl_half'], step_ops['afilt3d_half'],
                        obl_map_op, N, dZ_step, params.dT)
                else:
                    field = march_step_split_kt(
                        field, step_ops['HH_half'], step_ops['abl_half'], step_ops['afilt3d_half'],
                        N, dZ_step, params.dT)
            elif params.useTVD:
                norm_step = dZ_step / (c0 / params.f0)
                beta_tvd = max(1.0, 2.0 - params.adaptiveFilterStrength * norm_step * 10)
                if use_obl_nl:
                    field = march_step_split_tvd_obl(
                        field, step_ops['HH_half'], step_ops['abl_half'], step_ops['afilt3d_half'],
                        obl_map_op, N, dZ_step, params.dT, beta_tvd)
                else:
                    field = march_step_split_tvd(
                        field, step_ops['HH_half'], step_ops['abl_half'], step_ops['afilt3d_half'],
                        N, dZ_step, params.dT, beta_tvd)
            else:
                if use_obl_nl:
                    field = march_step_split_standard_obl(
                        field, step_ops['HH_half'], step_ops['abl_half'], step_ops['afilt3d_half'],
                        obl_map_op, N, dZ_step, params.dT)
                else:
                    field = march_step_split_standard(
                        field, step_ops['HH_half'], step_ops['abl_half'], step_ops['afilt3d_half'],
                        N, dZ_step, params.dT)
        else:
            if use_obl_nl:
                field = march_step_sequential_obl(
                    field, step_ops['HH'], step_ops['abl'], step_ops['afilt3d'],
                    obl_map_op, N, dZ_step, params.dT)
            else:
                field = march_step_sequential(
                    field, step_ops['HH'], step_ops['abl'], step_ops['afilt3d'],
                    N, dZ_step, params.dT)

        # --- enhanced boundary treatments (applied after each march step) ---
        if params.useFreqWeightedBoundary:
            field = _freq_weighted_boundary_step(field, ops['abl_freq'])

        if params.useSuperAbsorbing:
            field = _super_absorbing_step(
                field, ops['bdy_weight_x'], ops['bdy_weight_y'],
                params.dX, params.dY, params.dT, c0,
                params.superAbsorbingStrength)

        # --- phase screens for heterogeneous propagation ---
        if params.phaseScreens is not None:
            z_current = sum(zvec)
            for screen in params.phaseScreens:
                # Screens can be 2-tuple (z, phase) or 3-tuple (z, phase, amp)
                z_ps = screen[0]
                ps_array = screen[1]
                amp_array = screen[2] if len(screen) > 2 else None
                # Apply screen when we cross its z-position
                if z_current < z_ps <= z_current + dZ_step:
                    ps_jax = jnp.array(ps_array, dtype=jnp.float32)
                    amp_jax = jnp.array(amp_array, dtype=jnp.float32) if amp_array is not None else None
                    field = _apply_phase_screen(field, ps_jax, f0_bin, amp_jax)
                    if verbose:
                        print(f'  applied phase screen at z = {z_ps:.4f} m')

        # --- distributed source injection (bowl transducer) ---
        if params.sourcePlanes is not None:
            z_current = sum(zvec)
            for sp_z, sp_field in params.sourcePlanes:
                if z_current < sp_z <= z_current + dZ_step:
                    field = field + jnp.array(sp_field, dtype=jnp.float32)
                    if verbose:
                        print(f'  injected source plane at z = {sp_z*1e3:.2f} mm')

        field_np_after = np.array(field)
        I_after = np.sum(field_np_after ** 2, axis=2)
        pIloss[:, :, cc] = np.maximum(0, I_before - I_after)
        pI[:, :, cc] = I_after
        pnp[:, :, cc] = np.min(field_np_after, axis=2)
        ppp[:, :, cc] = np.max(field_np_after, axis=2)
        pax[:, cc] = field_np_after[nX // 2, nY // 2, :]
        zvec.append(dZ_step)

        if _tof_enabled:
            if _tof_use_mf:
                tof[:, :, cc] = _extract_tof_mf(
                    field_np_after, _tof_t, sum(zvec), c0,
                    _tof_ref, _tof_t_ref_peak)
            else:
                tof[:, :, cc] = _extract_tof(
                    field_np_after, _tof_t, sum(zvec), c0, tof_env_ratio)

        if per_step_callback is not None:
            per_step_callback(cc, sum(zvec), field)

        cc += 1

    elapsed = _time.time() - t0
    if verbose:
        print(f'Simulation completed in {elapsed:.2f} s  ({cc} steps)')

    # --- trim outputs ---
    pnp = pnp[:, :, :cc]
    ppp = ppp[:, :, :cc]
    pI = pI[:, :, :cc]
    pIloss = pIloss[:, :, :cc]
    pax = pax[:, :cc]
    zaxis = np.cumsum(zvec[:cc])

    if _tof_enabled:
        tof = tof[:, :, :cc]
        return np.array(field), pnp, ppp, pI, pIloss, zaxis, pax, tof
    return np.array(field), pnp, ppp, pI, pIloss, zaxis, pax
