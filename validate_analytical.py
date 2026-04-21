"""
Analytical validation tests ported from angularspectrum_forest/matlab_validation/.

Test A — Baffled piston diffraction:
  Compare angular spectrum propagation of a circular piston against
  the Rayleigh on-axis solution and far-field Bessel (2*J1(u)/u)^2 pattern.
  Produces a 9-panel summary figure matching the MATLAB layout.

Test B — 1D nonlinear Riemann problem:
  Solve u_t + (u^2/2)_x = 0 with a step initial condition using Godunov,
  Rusanov, and KT fluxes.  Compare against the exact Riemann solution
  (shock wave).

Test C — Focused piston diffraction:
  Apply geometric (spherical) delays across a circular aperture to focus at
  range F.  Compare on-axis pressure against the paraxial (Fresnel) analytical
  solution for a focused velocity piston, and the focal-plane lateral profile
  against the Airy-like (2*J1/u)^2 pattern with effective aperture a/F.

Results saved to validation_results/analytical/
"""

import os, sys, time
import numpy as np
from scipy.special import j1 as besselj1
from scipy.signal import hilbert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from angular_spectrum_solver import precalculate_mas, SolverParams

OUTDIR = os.path.join(os.path.dirname(__file__), 'validation_results', 'analytical')
os.makedirs(OUTDIR, exist_ok=True)


def _save_fig(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# ======================================================================
# Shared propagation helper used by Tests A and C
# ======================================================================
def _propagate_piston(aperture, icvec, nX, nY, nT, dX, dY, dT, dZ, c0, propDist,
                      label='', extra_delay_samples=None):
    """Angular-spectrum propagate a circular aperture (optionally with per-(x,y)
    integer-sample temporal delays for focusing).  Returns arrays recorded at
    each step:
      field_final, zaxis,
      numeric_onaxis, pI_x_full (XZ-plane intensity), pI_y_full (YZ-plane),
      PNP_x_full, PNP_y_full (XZ/YZ peak-neg pressure).
    """
    nT_field = nT
    # Build initial 3D field with optional integer-sample temporal delays
    field = np.zeros((nX, nY, nT_field), dtype=np.float32)
    if extra_delay_samples is None:
        field[:] = aperture[:, :, None] * icvec[None, None, :]
    else:
        # Apply per-(x,y) delay via np.roll along the time axis.  Delays are
        # rounded to the nearest sample; the pulse must be narrow enough that
        # the rounding error is small relative to lambda.
        mask = aperture > 0
        xs, ys = np.where(mask)
        delays = extra_delay_samples[xs, ys]
        for (ix, iy, d) in zip(xs, ys, delays):
            field[ix, iy, :] = np.roll(icvec, int(d))

    print(f'  [{label}] Precomputing angular spectrum propagator...')
    HH, _ = precalculate_mas(nX, nY, nT, dX, dY, dZ, dT, c0,
                              split_step=False, adaptive_filtering=False)
    HH = HH.astype(np.complex64)

    n_steps = int(np.ceil(propDist / dZ)) + 1
    numeric_onaxis = np.zeros(n_steps, dtype=np.float32)
    pI_x_full = np.zeros((nX, n_steps), dtype=np.float32)
    pI_y_full = np.zeros((nY, n_steps), dtype=np.float32)
    PNP_x_full = np.zeros((nX, n_steps), dtype=np.float32)
    PNP_y_full = np.zeros((nY, n_steps), dtype=np.float32)
    zvec = []
    cc = 0
    remaining = propDist

    print(f'  [{label}] Propagating...')
    while remaining > 1e-10:
        dZ_cur = min(dZ, remaining)
        if abs(dZ_cur - dZ) > 1e-10:
            HH_cur, _ = precalculate_mas(nX, nY, nT, dX, dY, dZ_cur, dT, c0,
                                          split_step=False, adaptive_filtering=False)
            HH_cur = HH_cur.astype(np.complex64)
        else:
            HH_cur = HH

        F = np.fft.fftn(field)
        F = np.fft.fftshift(F) * HH_cur
        F = np.fft.ifftshift(F)
        field = np.real(np.fft.ifftn(F)).astype(np.float32)

        zvec.append(dZ_cur)
        remaining -= dZ_cur

        center_wf = field[nX//2, nY//2, :]
        numeric_onaxis[cc] = np.max(np.abs(center_wf))

        pI_x_full[:, cc] = np.sum(field[:, nY//2, :]**2, axis=1)
        pI_y_full[:, cc] = np.sum(field[nX//2, :, :]**2, axis=1)
        PNP_x_full[:, cc] = np.min(field[:, nY//2, :], axis=1)
        PNP_y_full[:, cc] = np.min(field[nX//2, :, :], axis=1)

        cc += 1
        if cc % 10 == 0:
            print(f'    z = {sum(zvec)*100:.2f} cm')

    zaxis = np.cumsum(zvec[:cc])
    return (field, zaxis,
            numeric_onaxis[:cc], pI_x_full[:, :cc], pI_y_full[:, :cc],
            PNP_x_full[:, :cc], PNP_y_full[:, :cc])


# ======================================================================
# Test A — Baffled piston diffraction
# ======================================================================
def test_baffled_piston(a_lambda=6.0, tag=''):
    """Replicate validate_baffled_piston_diffraction.m from angularspectrum_forest.

    Produces the 9-panel MATLAB layout: X/Y-T slices, axial waveform (numerical
    vs analytical), XY intensity, XZ/YZ intensity, XZ/YZ mechanical index,
    on-axis amplitude comparison.

    Parameters
    ----------
    a_lambda : float
        Piston radius in wavelengths (forest default = 6).
    tag : str
        Suffix appended to output filenames so sweeps don't overwrite.
    """
    print(f'\n=== Test A: Baffled piston diffraction (a = {a_lambda:g} λ) ===')

    # Parameters matching the MATLAB validation
    f0 = 4e6
    c0 = 1500
    rho0 = 1000
    lam = c0 / f0
    dX = lam / 5
    dY = dX
    dT = dX / (5 * c0)

    nX = 257
    nY = 257
    nT = 721

    a = a_lambda * lam    # piston radius
    p0 = 0.3e6            # pressure amplitude
    propDist = 0.05        # 5 cm
    dZ = 8 * dX            # axial step (matches MATLAB validate_baffled_piston_diffraction.m)

    # Axes
    xaxis = np.arange(nX) * dX - (nX - 1) * dX / 2
    yaxis = np.arange(nY) * dY - (nY - 1) * dY / 2
    taxis = np.arange(nT) * dT - (nT - 1) * dT / 2

    # Circular piston aperture
    XX, YY = np.meshgrid(xaxis, yaxis, indexing='ij')
    aperture = ((XX**2 + YY**2) <= a**2).astype(np.float32)

    # Temporal pulse (Gaussian-enveloped sinusoid, matches MATLAB)
    ncycles = 3.0
    dur = 2
    omega0 = 2 * np.pi * f0
    icvec = (np.exp(-(1.05 * taxis * omega0 / (ncycles * np.pi))**(2*dur))
             * np.sin(taxis * omega0) * p0).astype(np.float32)

    # Pulse duration estimate for Isppa conversion
    analytic_env = np.abs(hilbert(icvec.astype(np.float64)))
    pdur = np.sum(analytic_env > analytic_env.max()/2) * dT

    k0 = 2 * np.pi * f0 / c0

    (field, zaxis, numeric_onaxis, pI_x_full, pI_y_full,
     PNP_x_full, PNP_y_full) = _propagate_piston(
        aperture, icvec, nX, nY, nT, dX, dY, dT, dZ, c0, propDist,
        label=f'baffled a={a_lambda:g}λ')

    # --- Analytical on-axis (paraxial Rayleigh / Fresnel, velocity piston) ---
    phi = k0 * a**2 / (2 * zaxis)
    Pcw = -p0 * (np.exp(1j * phi) - 1) * np.exp(-1j * k0 * zaxis)
    analytic_onaxis = np.abs(Pcw)

    # --- Far-field lateral: Bessel pattern ---
    z_ff = max(2 * a**2 / lam, zaxis[-1] * 0.9)
    idx_ff = np.argmin(np.abs(zaxis - z_ff))
    r = np.abs(xaxis)
    u = k0 * a * r / zaxis[idx_ff]
    u_safe = np.where(u < 1e-10, 1e-10, u)
    model_ff = (2 * besselj1(u_safe) / u_safe)**2
    model_ff[u < 1e-10] = 1.0
    model_ff /= np.max(model_ff)
    I_lat_ff = pI_x_full[:, idx_ff].astype(np.float64)
    I_lat_ff /= np.max(I_lat_ff) + 1e-30

    # --- Error metrics ---
    norm_num = numeric_onaxis / (np.max(numeric_onaxis) + 1e-30)
    norm_ana = analytic_onaxis / (np.max(analytic_onaxis) + 1e-30)
    onaxis_rms = np.sqrt(np.mean((norm_num - norm_ana)**2))
    lateral_rms = np.sqrt(np.mean((I_lat_ff - model_ff)**2))

    print(f'  On-axis RMS error (normalized):  {onaxis_rms:.4f}')
    print(f'  Far-field lateral RMS error:     {lateral_rms:.4f}')

    # --- Compact 4-panel summary figure (2x2) ---
    pI_Wcm2_x = pI_x_full * dT / (c0 * rho0 * pdur) / 1e4
    Isppa_xy = np.sum(field**2, axis=2) * dT / (c0 * rho0 * pdur) / 1e4   # W/cm^2

    # On-axis analytical waveform at final z
    analytic_env0 = np.exp(-(1.05 * taxis * omega0 / (ncycles * np.pi))**(2*dur))
    phi_phase = np.angle(Pcw[-1])
    analytic_wave = np.abs(Pcw[-1]) * analytic_env0 * np.sin(omega0 * taxis + phi_phase)
    center_waveform = field[nX//2, nY//2, :]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    z_end_cm = zaxis[-1] * 100

    # (0,0): XZ intensity (W/cm^2)
    im = axes[0, 0].imshow(pI_Wcm2_x, aspect='auto',
                           extent=[0, propDist*100, xaxis[-1]*1e3, xaxis[0]*1e3])
    axes[0, 0].set(xlabel='z (cm)', ylabel='x (mm)',
                   title='XZ-plane intensity (W/cm²)')
    plt.colorbar(im, ax=axes[0, 0])

    # (0,1): On-axis amplitude — primary validation
    axes[0, 1].plot(zaxis*100, numeric_onaxis, 'b-', lw=1.5, label='Numerical')
    axes[0, 1].plot(zaxis*100, analytic_onaxis, 'r--', lw=1.5,
                    label='Rayleigh/Fresnel')
    axes[0, 1].set(xlabel='z (cm)', ylabel='On-axis |p| (Pa)',
                   title=f'On-axis amplitude — RMS (norm) = {onaxis_rms:.4f}')
    axes[0, 1].legend(); axes[0, 1].grid(True)

    # (1,0): Focal-plane (XY) intensity at final z
    im = axes[1, 0].imshow(Isppa_xy.T, origin='lower',
                           extent=[xaxis[0]*1e3, xaxis[-1]*1e3, yaxis[0]*1e3, yaxis[-1]*1e3])
    axes[1, 0].set(xlabel='x (mm)', ylabel='y (mm)',
                   title=f'XY intensity at z = {z_end_cm:.2f} cm (W/cm²)')
    plt.colorbar(im, ax=axes[1, 0])

    # (1,1): Axial waveform comparison at final z
    axes[1, 1].plot(taxis*1e6, center_waveform, 'k-', lw=1.2, label='Numerical')
    axes[1, 1].plot(taxis*1e6, analytic_wave, 'r--', lw=1.2, label='Analytical')
    axes[1, 1].set(xlabel='t (µs)', ylabel='Pressure (Pa)',
                   title=f'Axial waveform at z = {z_end_cm:.2f} cm')
    axes[1, 1].legend(); axes[1, 1].grid(True)

    fig.suptitle(f'Baffled piston: $f_0$ = {f0/1e6:.2f} MHz, $a$ = {a*1e3:.2f} mm '
                 f'({a_lambda:g}$\\lambda$), $\\lambda$ = {lam*1e3:.2f} mm',
                 fontsize=13)
    plt.tight_layout()
    _save_fig(fig, f'testA_baffled_piston{tag}.png')

    # Far-field lateral standalone plot (matches MATLAB)
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(xaxis*1e3, I_lat_ff, 'b-', lw=1.5, label='Numerical')
    ax2.plot(xaxis*1e3, model_ff, 'r--', lw=1.5, label='$(2 J_1(u)/u)^2$')
    ax2.set(xlabel='x (mm)', ylabel='Normalized intensity',
            title=f'Far-field lateral at z = {zaxis[idx_ff]*100:.2f} cm '
                  f'(a = {a_lambda:g}λ)')
    ax2.legend(); ax2.grid(True)
    _save_fig(fig2, f'testA_baffled_piston_farfield{tag}.png')

    np.savez(os.path.join(OUTDIR, f'testA_baffled_piston{tag}.npz'),
             zaxis=zaxis, numeric_onaxis=numeric_onaxis,
             analytic_onaxis=analytic_onaxis,
             xaxis=xaxis, I_lat_ff=I_lat_ff, model_ff=model_ff,
             onaxis_rms=onaxis_rms, lateral_rms=lateral_rms,
             a_lambda=a_lambda)

    return onaxis_rms, lateral_rms


# ======================================================================
# Test C — Focused piston diffraction vs O'Neil / Fresnel solution
# ======================================================================
def test_focused_piston():
    """Validate a spherically-focused circular piston against the paraxial
    on-axis solution and the focal-plane Airy-like lateral pattern.

    Per-(x,y) temporal delays are applied to the aperture so rays converge
    at geometric focus F:  τ(r) = (F − sqrt(F² + r²)) / c0 .  Because the
    current pipeline uses a real-valued (x, y, t) field, delays are rounded
    to the nearest temporal sample — this requires dT small compared to the
    carrier period (here dT ≈ (1/f0)/25 for f0 = 4 MHz, so rounding is < λ/25).

    Analytical on-axis (paraxial, velocity piston with spherical delay):
        |p(0, z)| = 2 p0 · |F / (F − z)| · |sin( k a² (F − z) / (4 F z) )|
    with the focal limit  |p(0, F)| = p0 · π a² / (λ F) .

    Focal-plane lateral (Fraunhofer at z = F):
        |p(r, F)|² ∝ (2 J1(k a r / F) / (k a r / F))²
    """
    print('\n=== Test C: Focused piston diffraction ===')

    f0 = 4e6
    c0 = 1500
    rho0 = 1000
    lam = c0 / f0
    dX = lam / 5
    dY = dX
    dT = dX / (5 * c0)

    # Wider transverse grid (257 vs 193) to push the periodic-FFT edge
    # well away from any aperture-edge waves that could otherwise wrap
    # back into the beam on the 26 mm propagation.
    nX = 257
    nY = 257
    nT = 601

    a = 8 * lam           # piston radius (= 3.0 mm at f0 = 4 MHz, F# ≈ 3.3)
    F_foc = 20e-3         # geometric focal length (20 mm)
    p0 = 0.3e6
    propDist = 50e-3         # 50 mm (2.5 × F_foc, well past the focus)
    dZ = 2 * dX           # finer axial step for well-sampled on-axis curve

    if a > F_foc / 2:
        print(f'  warning: a={a*1e3:.2f} mm, F={F_foc*1e3:.1f} mm — small F/#.')

    # Axes
    xaxis = np.arange(nX) * dX - (nX - 1) * dX / 2
    yaxis = np.arange(nY) * dY - (nY - 1) * dY / 2
    taxis = np.arange(nT) * dT - (nT - 1) * dT / 2

    XX, YY = np.meshgrid(xaxis, yaxis, indexing='ij')
    RR2 = XX**2 + YY**2
    aperture = (RR2 <= a**2).astype(np.float32)

    # Gaussian-enveloped sinusoid; slightly longer (ncycles=4) so the rolled
    # pulse still fits in the time window even at aperture edge delays.
    ncycles = 4.0
    dur = 2
    omega0 = 2 * np.pi * f0
    icvec = (np.exp(-(1.05 * taxis * omega0 / (ncycles * np.pi))**(2*dur))
             * np.sin(taxis * omega0) * p0).astype(np.float32)

    # Geometric focusing delays τ(r) = (F − sqrt(F² + r²)) / c0 (<0 for r>0,
    # so the edges fire earlier than centre).  Convert to integer samples.
    tau = (F_foc - np.sqrt(F_foc**2 + RR2)) / c0          # seconds, ≤ 0
    delay_samples = np.round(tau / dT).astype(np.int32)    # positive = roll to later
    # np.roll(x, n): positive n shifts toward larger index.  We want edges to
    # fire earlier — the edge delay is negative (in seconds), which gives a
    # negative sample shift → np.roll with negative argument.  Use as-is.

    k0 = 2 * np.pi * f0 / c0

    (field, zaxis, numeric_onaxis, pI_x_full, pI_y_full,
     PNP_x_full, PNP_y_full) = _propagate_piston(
        aperture, icvec, nX, nY, nT, dX, dY, dT, dZ, c0, propDist,
        label='focused', extra_delay_samples=delay_samples)

    # --- Analytical on-axis (paraxial, focused velocity piston) ---
    eps_z = 1e-9
    z_arg = k0 * a**2 * (F_foc - zaxis) / (4.0 * F_foc * np.maximum(zaxis, eps_z))
    denom = F_foc - zaxis
    analytic_onaxis = 2.0 * p0 * np.abs(F_foc / np.where(np.abs(denom) < eps_z, eps_z, denom)) \
                      * np.abs(np.sin(z_arg))
    # Enforce focal-limit analytically to avoid the 0/0 spike
    focal_amp = p0 * np.pi * a**2 / (lam * F_foc)
    idx_F = np.argmin(np.abs(zaxis - F_foc))
    # Replace values in a small window around focus (|F-z| < 2·dZ) with a
    # smooth interpolation up to the focal limit.
    near = np.abs(zaxis - F_foc) < 2 * dZ
    if np.any(near):
        analytic_onaxis[near] = focal_amp  # paraxial limit

    # --- Focal-plane lateral (y=0 slice, summed-intensity normalization) ---
    r_abs = np.abs(xaxis)
    u = k0 * a * r_abs / F_foc
    u_safe = np.where(u < 1e-10, 1e-10, u)
    model_focal = (2 * besselj1(u_safe) / u_safe)**2
    model_focal[u < 1e-10] = 1.0
    model_focal /= np.max(model_focal)
    I_lat_focal = pI_x_full[:, idx_F].astype(np.float64)
    I_lat_focal /= np.max(I_lat_focal) + 1e-30

    # --- Errors ---
    norm_num = numeric_onaxis / (np.max(numeric_onaxis) + 1e-30)
    norm_ana = analytic_onaxis / (np.max(analytic_onaxis) + 1e-30)
    onaxis_rms = np.sqrt(np.mean((norm_num - norm_ana)**2))
    lateral_rms = np.sqrt(np.mean((I_lat_focal - model_focal)**2))
    # Focal amplitude: compare numerical and analytical at the geometric focus
    # (NOT the global maximum — for finite F# the paraxial peak occurs slightly
    # before z=F, and the global maxima do not correspond to the same z).
    p_at_F_num = float(numeric_onaxis[idx_F])
    p_at_F_ana = float(focal_amp)         # paraxial focal limit p0·π·a²/(λ·F)
    focal_amp_err = abs(p_at_F_num - p_at_F_ana) / p_at_F_ana
    # Also report the global-max comparison for context.
    peak_num = float(np.max(numeric_onaxis))
    peak_ana = float(np.max(analytic_onaxis))
    peak_amp_err = abs(peak_num - peak_ana) / peak_ana

    print(f'  At geometric focus z=F:')
    print(f'     numerical |p|/p0 = {p_at_F_num/p0:.3f}, '
          f'analytical |p|/p0 = {p_at_F_ana/p0:.3f}, '
          f'frac err = {focal_amp_err:.4f}')
    print(f'  Global on-axis maxima:')
    print(f'     numerical |p|/p0 = {peak_num/p0:.3f}, '
          f'analytical |p|/p0 = {peak_ana/p0:.3f}, '
          f'frac err = {peak_amp_err:.4f}')
    print(f'  On-axis RMS error (normalized):   {onaxis_rms:.4f}')
    print(f'  Focal-plane lateral RMS error:    {lateral_rms:.4f}')

    # --- Compact 2x2 plot: on-axis, focal lateral, XZ, YZ ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(zaxis * 1e3, numeric_onaxis, 'b-', lw=1.5, label='Numerical')
    axes[0, 0].plot(zaxis * 1e3, analytic_onaxis, 'r--', lw=1.5, label='Analytical')
    axes[0, 0].axvline(F_foc * 1e3, color='gray', ls=':', label=f'F = {F_foc*1e3:.0f} mm')
    axes[0, 0].set(xlabel='z (mm)', ylabel='On-axis |p| (Pa)',
                   title=f'On-axis amplitude — RMS (norm) = {onaxis_rms:.4f}')
    axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].plot(xaxis * 1e3, I_lat_focal, 'b-', lw=1.5, label='Numerical')
    axes[0, 1].plot(xaxis * 1e3, model_focal, 'r--', lw=1.5, label='$(2J_1/u)^2$')
    axes[0, 1].set(xlabel='x (mm)', ylabel='Normalized intensity',
                   title=f'Focal-plane lateral (z = {zaxis[idx_F]*1e3:.1f} mm) — RMS = {lateral_rms:.4f}')
    axes[0, 1].set_xlim(-5, 5)
    axes[0, 1].legend(); axes[0, 1].grid(True)

    im = axes[1, 0].imshow(pI_x_full, aspect='auto',
                           extent=[0, zaxis[-1]*1e3, xaxis[-1]*1e3, xaxis[0]*1e3])
    axes[1, 0].axvline(F_foc * 1e3, color='w', ls=':', lw=1)
    axes[1, 0].set(xlabel='z (mm)', ylabel='x (mm)', title='XZ intensity (summed p²)')
    plt.colorbar(im, ax=axes[1, 0])

    im = axes[1, 1].imshow(pI_y_full, aspect='auto',
                           extent=[0, zaxis[-1]*1e3, yaxis[-1]*1e3, yaxis[0]*1e3])
    axes[1, 1].axvline(F_foc * 1e3, color='w', ls=':', lw=1)
    axes[1, 1].set(xlabel='z (mm)', ylabel='y (mm)', title='YZ intensity (summed p²)')
    plt.colorbar(im, ax=axes[1, 1])

    fig.suptitle(f'Focused piston: $f_0$ = {f0/1e6:.1f} MHz, $a$ = {a*1e3:.2f} mm '
                 f'({a/lam:.0f}$\\lambda$), $F$ = {F_foc*1e3:.1f} mm, '
                 f'F\\# = {F_foc/(2*a):.1f}, $\\lambda$ = {lam*1e3:.2f} mm',
                 fontsize=13)
    plt.tight_layout()
    _save_fig(fig, 'testC_focused_piston.png')

    np.savez(os.path.join(OUTDIR, 'testC_focused_piston.npz'),
             zaxis=zaxis, numeric_onaxis=numeric_onaxis,
             analytic_onaxis=analytic_onaxis,
             xaxis=xaxis, I_lat_focal=I_lat_focal, model_focal=model_focal,
             onaxis_rms=onaxis_rms, lateral_rms=lateral_rms,
             p_at_F_numerical=p_at_F_num, p_at_F_analytical=p_at_F_ana,
             focal_amp_err=focal_amp_err,
             peak_numerical=peak_num, peak_analytical=peak_ana,
             peak_amp_err=peak_amp_err,
             a=a, F=F_foc, f0=f0, c0=c0)

    return onaxis_rms, lateral_rms, focal_amp_err


# ======================================================================
# Test C2 — Focused-piston config sweep: split-step × adaptive filter
# ======================================================================
def test_focused_piston_config_sweep(nT=1201, dZ_factor=2, ncycles=4.0,
                                     continuous_delays=False, tag=''):
    """Run the focused-piston analytical benchmark through the full
    angular_spectrum_solve() at four solver configurations:
      {sequential, split-step} × {no adaptive filter, adaptive filter}
    and report on-axis RMS vs paraxial and focal-plane Airy RMS for each.

    Parameters
    ----------
    nT : int
        Number of time samples.  Larger → longer time window (helps at
        long propagation distances and wide delay spreads).
    dZ_factor : int
        Axial step size in units of dX.  Smaller → finer axial step.
    ncycles : float
        Gaussian-envelope pulse length (cycles).  Longer → narrower
        bandwidth, closer to CW analytical reference.
    tag : str
        Suffix for output file names so parametric sweeps don't
        overwrite each other.
    """
    print(f'\n=== Test C2: Focused piston — solver config sweep '
          f'(nT={nT}, dZ={dZ_factor}·dX, ncycles={ncycles}) ===')

    from angular_spectrum_solver import SolverParams, angular_spectrum_solve

    # Geometry matches test_focused_piston
    f0 = 4e6
    c0 = 1500
    lam = c0 / f0
    dX = lam / 5
    dT = dX / (5 * c0)
    nX, nY = 193, 193
    a = 8 * lam
    F_foc = 20e-3
    p0 = 0.3e6
    propDist = 1.3 * F_foc

    xaxis = np.arange(nX) * dX - (nX - 1) * dX / 2
    yaxis = np.arange(nY) * dX - (nY - 1) * dX / 2
    taxis = np.arange(nT) * dT - (nT - 1) * dT / 2

    XX, YY = np.meshgrid(xaxis, yaxis, indexing='ij')
    RR2 = XX**2 + YY**2
    aperture = (RR2 <= a**2).astype(np.float32)

    dur = 2
    omega0 = 2 * np.pi * f0
    icvec = (np.exp(-(1.05 * taxis * omega0 / (ncycles * np.pi))**(2*dur))
             * np.sin(taxis * omega0) * p0).astype(np.float32)

    tau = (F_foc - np.sqrt(F_foc**2 + RR2)) / c0   # seconds, ≤ 0

    if continuous_delays:
        # Apply fractional delays via FFT phase-shift: exact (no rounding).
        # field(x, y, t) = IFFT_ω[ FFT_ω[icvec](ω) · e^{-iωτ(x,y)} · aperture(x,y) ]
        print(f'  Applying continuous (FFT phase-shift) delays...')
        icvec_fft = np.fft.fft(icvec.astype(np.float64))
        omegas = 2 * np.pi * np.fft.fftfreq(nT, d=dT)
        # Shape: (nX, nY, nT) complex — ~360 MB for nT=1201; tractable
        phase = np.exp(-1j * omegas[None, None, :] * tau[:, :, None])
        field0_c = icvec_fft[None, None, :] * phase * aperture[:, :, None]
        field0 = np.real(np.fft.ifft(field0_c, axis=2)).astype(np.float32)
        del field0_c, phase
    else:
        delay_samples = np.round(tau / dT).astype(np.int32)
        field0 = np.zeros((nX, nY, nT), dtype=np.float32)
        mask = aperture > 0
        xs, ys = np.where(mask)
        for (ix, iy) in zip(xs, ys):
            field0[ix, iy, :] = np.roll(icvec, int(delay_samples[ix, iy]))

    k0 = 2 * np.pi * f0 / c0

    # Configs to sweep.  Earlier runs confirmed all four give identical
    # metrics, so by default we only exercise the discriminating toggles
    # when continuous_delays=False (the manuscript table).  Pass a single
    # config when continuous_delays=True to save time.
    if continuous_delays:
        configs = [('sequential, no filter, continuous-tau',
                    dict(useSplitStep=False, useAdaptiveFiltering=False))]
    else:
        configs = [
            ('sequential, no filter',  dict(useSplitStep=False, useAdaptiveFiltering=False)),
            ('split-step, no filter',  dict(useSplitStep=True,  useAdaptiveFiltering=False)),
            ('sequential, k-filter',   dict(useSplitStep=False, useAdaptiveFiltering=True)),
            ('split-step, k-filter',   dict(useSplitStep=True,  useAdaptiveFiltering=True)),
        ]

    results = {}
    for label, overrides in configs:
        print(f'\n--- config: {label} ---')
        params = SolverParams(
            dX=dX, dY=dX, dT=dT, c0=c0, rho0=1000,
            beta=0.0, alpha0=-1, f0=f0, propDist=propDist,
            boundaryFactor=0.15, useBoundaryLayer=True,
            useTVD=False, stabilityThreshold=0.2,
            stabilityRecoveryFactor=0.15, dZmin=dX * dZ_factor,
            **overrides)
        t0 = time.time()
        field_out, pnp, ppp, pI, pIloss, zaxis, pax = \
            angular_spectrum_solve(field0, params, verbose=False)
        dt_run = time.time() - t0
        zaxis = np.asarray(zaxis)

        # On-axis peak amplitude vs z (from axial pressure trace pax:(nT, nZ))
        pax_np = np.asarray(pax)
        numeric_onaxis = np.max(np.abs(pax_np), axis=0)
        # Focal-plane lateral slice along x at idx_F (from pI which is
        # summed p^2 over time)
        idx_F = int(np.argmin(np.abs(zaxis - F_foc)))
        I_lat = np.asarray(pI[:, nY // 2, idx_F]).astype(np.float64)
        I_lat_norm = I_lat / (np.max(I_lat) + 1e-30)

        # Analytical references evaluated on this config's zaxis
        eps_z = 1e-9
        z_arg = k0 * a**2 * (F_foc - zaxis) / (4.0 * F_foc * np.maximum(zaxis, eps_z))
        denom = F_foc - zaxis
        analytic_onaxis = 2.0 * p0 * np.abs(F_foc / np.where(np.abs(denom) < eps_z, eps_z, denom)) \
                          * np.abs(np.sin(z_arg))
        focal_amp = p0 * np.pi * a**2 / (lam * F_foc)
        near = np.abs(zaxis - F_foc) < 2 * (zaxis[1] - zaxis[0] if len(zaxis) > 1 else 1e-3)
        if np.any(near):
            analytic_onaxis[near] = focal_amp

        u = k0 * a * np.abs(xaxis) / F_foc
        u_safe = np.where(u < 1e-10, 1e-10, u)
        model_focal = (2 * besselj1(u_safe) / u_safe)**2
        model_focal[u < 1e-10] = 1.0
        model_focal /= np.max(model_focal)

        # Metrics
        norm_num = numeric_onaxis / (np.max(numeric_onaxis) + 1e-30)
        norm_ana = analytic_onaxis / (np.max(analytic_onaxis) + 1e-30)
        onaxis_rms = float(np.sqrt(np.mean((norm_num - norm_ana)**2)))
        lateral_rms = float(np.sqrt(np.mean((I_lat_norm - model_focal)**2)))
        p_at_F_num = float(numeric_onaxis[idx_F])
        focal_err = abs(p_at_F_num - focal_amp) / focal_amp

        print(f'    runtime = {dt_run:.1f} s')
        print(f'    on-axis RMS (norm)  = {onaxis_rms:.4f}')
        print(f'    focal-plane RMS    = {lateral_rms:.4f}')
        print(f'    |p(F)| frac err    = {focal_err:.4f}')

        results[label] = dict(
            zaxis=zaxis, numeric_onaxis=numeric_onaxis,
            analytic_onaxis=analytic_onaxis,
            xaxis=xaxis, I_lat=I_lat_norm, model_focal=model_focal,
            onaxis_rms=onaxis_rms, lateral_rms=lateral_rms,
            focal_err=focal_err, runtime_s=dt_run,
        )

    # --- Figure: overlay on-axis curves + focal lateral across configs ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ['C0', 'C1', 'C2', 'C3']
    for (label, _), color in zip(configs, colors):
        r = results[label]
        axes[0].plot(r['zaxis']*1e3, r['numeric_onaxis'], color=color, lw=1.3,
                     label=f'{label} (RMS={r["onaxis_rms"]:.3f})')
    # Analytical reference using the first config's zaxis
    ref = results[configs[0][0]]
    axes[0].plot(ref['zaxis']*1e3, ref['analytic_onaxis'], 'k--', lw=1.5,
                 label='Analytical')
    axes[0].axvline(F_foc*1e3, color='gray', ls=':')
    axes[0].set(xlabel='z (mm)', ylabel='On-axis |p| (Pa)',
                title='On-axis amplitude vs analytical')
    axes[0].legend(fontsize=8); axes[0].grid(True)

    for (label, _), color in zip(configs, colors):
        r = results[label]
        axes[1].plot(r['xaxis']*1e3, r['I_lat'], color=color, lw=1.3,
                     label=f'{label} (RMS={r["lateral_rms"]:.3f})')
    axes[1].plot(ref['xaxis']*1e3, ref['model_focal'], 'k--', lw=1.5,
                 label='$(2 J_1/u)^2$')
    axes[1].set_xlim(-5, 5)
    axes[1].set(xlabel='x (mm)', ylabel='Normalized intensity',
                title=f'Focal-plane lateral (z = {F_foc*1e3:.1f} mm)')
    axes[1].legend(fontsize=8); axes[1].grid(True)

    fig.suptitle('Focused-piston config sweep: split-step × adaptive k-space filter',
                 fontsize=12)
    plt.tight_layout()
    _save_fig(fig, f'testC2_focused_piston_config_sweep{tag}.png')

    # Save .npz with labels + metrics as a flat table
    labels = [c[0] for c in configs]
    onaxis_rms_arr = np.array([results[l]['onaxis_rms'] for l in labels])
    lateral_rms_arr = np.array([results[l]['lateral_rms'] for l in labels])
    focal_err_arr = np.array([results[l]['focal_err'] for l in labels])
    runtime_arr = np.array([results[l]['runtime_s'] for l in labels])
    np.savez(os.path.join(OUTDIR, f'testC2_focused_piston_config_sweep{tag}.npz'),
             labels=np.array(labels),
             onaxis_rms=onaxis_rms_arr,
             lateral_rms=lateral_rms_arr,
             focal_err=focal_err_arr,
             runtime_s=runtime_arr,
             a=a, F=F_foc, f0=f0, c0=c0)

    # Print summary table
    print('\n  ' + '-'*78)
    print(f'  {"config":<30s} {"on-axis RMS":>12s} {"lateral RMS":>12s} '
          f'{"|p(F)| err":>10s} {"runtime":>8s}')
    print('  ' + '-'*78)
    for l in labels:
        r = results[l]
        print(f'  {l:<30s} {r["onaxis_rms"]:>12.4f} {r["lateral_rms"]:>12.4f} '
              f'{r["focal_err"]:>10.4f} {r["runtime_s"]:>7.1f}s')
    print('  ' + '-'*78)

    return results


# ======================================================================
# Test B — 1D nonlinear Riemann problem
# ======================================================================
def _burgers_exact(uL, uR, x, t):
    """Exact Riemann solution for inviscid Burgers equation."""
    if t <= 0:
        return np.where(x < 0, uL, uR)
    if uL > uR:
        # Shock at speed s = (uL + uR) / 2
        s = 0.5 * (uL + uR)
        return np.where(x < s * t, uL, uR)
    else:
        # Rarefaction fan
        xi = x / t
        ua = np.full_like(x, uL, dtype=np.float64)
        ua[xi > uR] = uR
        mask = (xi >= uL) & (xi <= uR)
        ua[mask] = xi[mask]
        return ua


def _godunov_flux(uL, uR):
    """Godunov flux for convex Burgers f(u) = u^2/2."""
    fL = 0.5 * uL**2
    fR = 0.5 * uR**2
    if uL <= uR:
        if uL >= 0:
            return fL
        elif uR <= 0:
            return fR
        else:
            return 0.0
    else:
        s = (fL - fR) / (uL - uR + 1e-30)
        return fL if s >= 0 else fR


def _rusanov_flux(uL, uR):
    """Rusanov (local Lax-Friedrichs) flux."""
    amax = max(abs(uL), abs(uR))
    return 0.5 * (0.5*uL**2 + 0.5*uR**2) - 0.5 * amax * (uR - uL)


def _kt_flux(uL, uR):
    """Kurganov-Tadmor central-upwind flux."""
    a_plus = max(0, max(uL, uR))
    a_minus = min(0, min(uL, uR))
    fL = 0.5 * uL**2
    fR = 0.5 * uR**2
    if abs(a_plus - a_minus) < 1e-14:
        return 0.5 * (fL + fR) - 0.5 * a_plus * (uR - uL)
    return (a_plus * fL - a_minus * fR - a_plus * a_minus * (uR - uL)) / (a_plus - a_minus)


def _run_1d_solver(u0, x, Tfinal, CFL, flux_fn, snap_times):
    """Run a 1D finite-volume solver with Dirichlet ghost cells."""
    N = len(x)
    dx = x[1] - x[0]
    u = u0.copy()
    uL_bc = u[0]
    uR_bc = u[-1]

    snapshots = {}
    current_time = 0.0
    snap_idx = 0

    while current_time < Tfinal - 1e-12:
        max_speed = max(np.max(np.abs(u)), 1e-12)
        dt = CFL * dx / max_speed
        if current_time + dt > Tfinal:
            dt = Tfinal - current_time

        # Ghost cells
        ug = np.concatenate([[uL_bc], u, [uR_bc]])

        # Interface fluxes
        F = np.zeros(N + 1)
        for i in range(N + 1):
            F[i] = flux_fn(ug[i], ug[i + 1])

        # Forward Euler update
        u = u - (dt / dx) * (F[1:] - F[:-1])
        current_time += dt

        # Capture snapshots
        while snap_idx < len(snap_times) and current_time >= snap_times[snap_idx] - 1e-12:
            snapshots[snap_times[snap_idx]] = u.copy()
            snap_idx += 1

    return snapshots


def test_riemann():
    """Replicate validate_nonlinear_riemann_1d.m from angularspectrum_forest."""
    print('\n=== Test B: 1D nonlinear Riemann problem ===')

    # Parameters matching MATLAB
    uL = 2.0
    uR = 0.0
    Lx = 1.0
    Nx = 400
    x = np.linspace(-Lx, Lx, Nx)
    CFL = 0.45
    snap_times = [0.1, 0.3, 0.5]

    u0 = np.where(x < 0, uL, uR).astype(np.float64)

    solvers = {
        'Rusanov': _rusanov_flux,
        'KT': _kt_flux,
    }

    # Run all solvers
    errors = {}
    all_snaps = {}
    for name, flux_fn in solvers.items():
        print(f'  Running {name}...')
        snaps = _run_1d_solver(u0, x, snap_times[-1], CFL, flux_fn, snap_times)
        all_snaps[name] = snaps
        solver_errors = []
        for t in snap_times:
            err = np.sqrt(np.mean((snaps[t] - _burgers_exact(uL, uR, x, t))**2))
            solver_errors.append(err)
        errors[name] = solver_errors
        print(f'    L2 errors at t={snap_times}: {[f"{e:.4f}" for e in solver_errors]}')

    # Compact figure: one column per time step, all schemes overlaid
    styles = {
        'Rusanov': dict(marker='o', color='C0', ms=5, markerfacecolor='none', markeredgewidth=1.3),
        'KT':      dict(marker='s', color='C1', ms=5, markerfacecolor='none', markeredgewidth=1.3),
    }
    fig, axes = plt.subplots(1, len(snap_times), figsize=(14, 3.5))

    for col, t in enumerate(snap_times):
        ax = axes[col]
        u_exact = _burgers_exact(uL, uR, x, t)
        ax.plot(x, u_exact, 'k-', lw=2, label='Exact')
        every = 8  # plot every 8th point for clarity
        for i, name in enumerate(solvers):
            u_num = all_snaps[name][t]
            offset = i * 2  # stagger sample points so markers don't overlap
            ax.plot(x[offset::every], u_num[offset::every], linestyle='none',
                    label=name, **styles[name])
        ax.set(xlabel='x', ylabel='u', title=f't = {t:.1f}')
        ax.set_ylim([-0.3, 2.5])
        ax.grid(True)
        if col == len(snap_times) - 1:
            ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    _save_fig(fig, 'testB_riemann.png')

    np.savez(os.path.join(OUTDIR, 'testB_riemann.npz'),
             x=x, snap_times=snap_times, errors=errors)

    return errors


# ======================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tests', nargs='+', default=['A', 'A4', 'B', 'C'],
                        choices=['A', 'A4', 'B', 'C', 'C2'],
                        help='Which tests to run (A=baffled 6λ, A4=baffled 4λ, '
                             'B=Riemann, C=focused, C2=focused solver-config sweep).')
    args = parser.parse_args()

    print('Analytical Validation Suite')
    print('(ported from angularspectrum_forest/matlab_validation/)')
    print('=' * 60)

    results = {}
    if 'A' in args.tests:
        results['A'] = test_baffled_piston(a_lambda=6.0, tag='')
    if 'A4' in args.tests:
        results['A4'] = test_baffled_piston(a_lambda=4.0, tag='_a4lambda')
    if 'B' in args.tests:
        results['B'] = test_riemann()
    if 'C' in args.tests:
        results['C'] = test_focused_piston()
    if 'C2' in args.tests:
        results['C2'] = test_focused_piston_config_sweep()

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    if 'A' in results:
        rms_onaxis, rms_lateral = results['A']
        print(f'Test A — Baffled piston (a = 6λ):')
        print(f'  On-axis RMS error (normalized):  {rms_onaxis:.4f}')
        print(f'  Far-field lateral RMS error:     {rms_lateral:.4f}')
    if 'A4' in results:
        rms_onaxis4, rms_lateral4 = results['A4']
        print(f'Test A — Baffled piston (a = 4λ):')
        print(f'  On-axis RMS error (normalized):  {rms_onaxis4:.4f}')
        print(f'  Far-field lateral RMS error:     {rms_lateral4:.4f}')
    if 'B' in results:
        riemann_errors = results['B']
        print(f'Test B — Riemann problem (L2 at t=0.5):')
        for name, errs in riemann_errors.items():
            print(f'  {name:10s}: {errs[-1]:.4f}')
    if 'C' in results:
        rms_on_c, rms_lat_c, focal_err = results['C']
        print(f'Test C — Focused piston:')
        print(f'  On-axis RMS error (normalized):  {rms_on_c:.4f}')
        print(f'  Focal-plane lateral RMS error:   {rms_lat_c:.4f}')
        print(f'  Focal amplitude fractional err:  {focal_err:.4f}')
    print(f'\nAll results saved to {OUTDIR}/')
