"""
Validation suite for angular_spectrum_solver.py

Tests:
  1. Linear propagation (beta=0): compare analytic diffraction pattern
  2. Nonlinear propagation: compare sequential vs split-step
  3. TVD limiter effect: measure total-variation reduction
  4. Adaptive filtering: check high-k suppression
  5. Energy conservation / attenuation-loss tracking
  6. Convergence: halve dZ and check error reduction
  9. Synthetic split linear+nonlinear order test for Strang+KT

Results are saved to validation_results/ as .npz and .png files.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax.numpy as jnp
from scipy.signal import hilbert

# -- path setup ---
sys.path.insert(0, os.path.dirname(__file__))
from angular_spectrum_solver import (
    SolverParams, angular_spectrum_solve,
    precalculate_mas, precalculate_abl, precalculate_ad_pow2, ablvec,
    _rusanov_flux_standard, _rusanov_flux_tvd, _kt_flux, _kt_flux_minmod,
    _kt_flux_adaptive,
)

OUTDIR = os.path.join(os.path.dirname(__file__), 'validation_results')
os.makedirs(OUTDIR, exist_ok=True)


# ======================================================================
# Helpers
# ======================================================================
def _make_ic(nX, nY, nT, dX, dT, f0, c0, p0, wX, wY, ncycles=1.5):
    """Create a simple rectangular-aperture pulsed initial condition."""
    xaxis = np.arange(nX) * dX - (nX - 1) * dX / 2
    yaxis = np.arange(nY) * dX - (nY - 1) * dX / 2
    omega0 = 2 * np.pi * f0
    dur = 2
    t = np.arange(nT) * dT
    t -= np.mean(t)

    # Aperture mask
    X, Y = np.meshgrid(xaxis, yaxis, indexing='ij')
    ap = np.where((np.abs(X) <= wX / 2) & (np.abs(Y) <= wY / 2), 1.0, 0.0)

    # Time signal
    icvec = np.exp(-(1.05 * t * omega0 / (ncycles * np.pi)) ** (2 * dur)) * np.sin(t * omega0) * p0
    apa = ap[:, :, np.newaxis] * icvec[np.newaxis, np.newaxis, :]
    return apa, xaxis, yaxis, t


def _total_variation(field_3d):
    """Total variation along axis 2."""
    return np.sum(np.abs(np.diff(field_3d, axis=2)))


def _save_fig(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# ======================================================================
# Test 1 — Linear propagation (beta=0)
# ======================================================================
def test_linear_propagation():
    print('\n=== Test 1: Linear propagation (beta=0) ===')
    f0 = 3e6;  c0 = 1500;  lam = c0 / f0
    dX = lam / 5;  dT = dX / (5 * c0)
    wX = 5e-3;  wY = 5e-3;  p0 = 1e3
    nX = 51;  nY = 51;  nT = 101

    apa, xaxis, yaxis, t = _make_ic(nX, nY, nT, dX, dT, f0, c0, p0, wX, wY)

    results = {}
    for label, split in [('sequential', False), ('split_step', True)]:
        params = SolverParams(
            dX=dX, dY=dX, dT=dT, c0=c0, rho0=1000, beta=0.0,
            alpha0=-1, f0=f0, propDist=2e-2, boundaryFactor=0.15,
            useSplitStep=split, useAdaptiveFiltering=False,
            stabilityThreshold=0.2, stabilityRecoveryFactor=0.15,
            dZmin=dX * 4)
        field, pnp, ppp, pI, pIloss, zaxis, pax = angular_spectrum_solve(apa, params, verbose=False)
        results[label] = dict(field=field, pI=pI, zaxis=zaxis)

    # Compare peak intensity profiles
    pI_seq = results['sequential']['pI']
    pI_ss = results['split_step']['pI']
    n_common = min(pI_seq.shape[2], pI_ss.shape[2])

    diff = np.max(np.abs(pI_seq[:, :, :n_common] - pI_ss[:, :, :n_common]))
    rel_diff = diff / (np.max(np.abs(pI_seq[:, :, :n_common])) + 1e-30)
    print(f'  max |pI_seq - pI_ss| = {diff:.4e}')
    print(f'  relative difference  = {rel_diff:.4e}')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    z_seq = results['sequential']['zaxis']
    z_ss = results['split_step']['zaxis']
    axes[0].plot(z_seq * 1e3, pI_seq[nX // 2, nY // 2, :], label='Sequential')
    axes[0].plot(z_ss * 1e3, pI_ss[nX // 2, nY // 2, :], '--', label='Split-step')
    axes[0].set(xlabel='z (mm)', ylabel='On-axis intensity', title='Linear: on-axis intensity')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].imshow(pI_seq[:, nY // 2, :], aspect='auto',
                   extent=[z_seq[0] * 1e3, z_seq[-1] * 1e3, xaxis[0] * 1e3, xaxis[-1] * 1e3])
    axes[1].set(xlabel='z (mm)', ylabel='x (mm)', title='Sequential: x-z intensity')

    axes[2].imshow(pI_ss[:, nY // 2, :], aspect='auto',
                   extent=[z_ss[0] * 1e3, z_ss[-1] * 1e3, xaxis[0] * 1e3, xaxis[-1] * 1e3])
    axes[2].set(xlabel='z (mm)', ylabel='x (mm)', title='Split-step: x-z intensity')

    _save_fig(fig, 'test1_linear.png')
    np.savez(os.path.join(OUTDIR, 'test1_linear.npz'),
             pI_seq=pI_seq, pI_ss=pI_ss, z_seq=z_seq, z_ss=z_ss,
             rel_diff=rel_diff)
    return rel_diff


# ======================================================================
# Test 2 — Nonlinear: sequential vs split-step
# ======================================================================
def test_nonlinear_comparison():
    print('\n=== Test 2: Nonlinear — Rusanov vs KT ===')
    f0 = 3e6;  c0 = 1500;  lam = c0 / f0
    dX = lam / 5;  dT = dX / (5 * c0)
    wX = 5e-3;  wY = 5e-3;  p0 = 0.03e6
    nX = 51;  nY = 51;  nT = 101

    apa, xaxis, yaxis, t = _make_ic(nX, nY, nT, dX, dT, f0, c0, p0, wX, wY)

    configs = [
        ('Rusanov',  dict(useSplitStep=True, fluxScheme='rusanov', useTVD=False)),
        ('KT',       dict(useSplitStep=True, fluxScheme='kt')),
    ]
    results = {}
    for label, kw in configs:
        params = SolverParams(
            dX=dX, dY=dX, dT=dT, c0=c0, rho0=1000, beta=3.5,
            alpha0=0.5, attenPow=1, f0=f0, propDist=3e-2,
            boundaryFactor=0.15,
            useAdaptiveFiltering=False,
            stabilityThreshold=0.2, stabilityRecoveryFactor=0.15,
            dZmin=dX * 4, **kw)
        field, pnp, ppp, pI, pIloss, zaxis, pax = angular_spectrum_solve(apa, params, verbose=False)
        results[label] = dict(field=field, pnp=pnp, ppp=ppp, pI=pI, zaxis=zaxis, pax=pax)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    styles = {'Rusanov': '-', 'KT': '--'}
    for label, _ in configs:
        ls = styles[label]
        r = results[label]
        z = r['zaxis']
        axes[0, 0].plot(z * 1e3, r['pI'][nX // 2, nY // 2, :], ls, label=label)
        axes[0, 1].plot(z * 1e3, r['ppp'][nX // 2, nY // 2, :], ls, label=f'p+ {label}')
        axes[0, 1].plot(z * 1e3, r['pnp'][nX // 2, nY // 2, :], ls, label=f'p- {label}')
        axes[1, 0].plot(t * 1e6, r['pax'][:, -1], ls, label=label)

    axes[0, 0].set(xlabel='z (mm)', ylabel='Intensity', title='On-axis intensity')
    axes[0, 0].legend();  axes[0, 0].grid(True)
    axes[0, 1].set(xlabel='z (mm)', ylabel='Pa', title='Peak pressures')
    axes[0, 1].legend();  axes[0, 1].grid(True)
    axes[1, 0].set(xlabel='t (us)', ylabel='Pa', title='Final axial waveform')
    axes[1, 0].legend();  axes[1, 0].grid(True)

    # Harmonic content comparison
    for label, _ in configs:
        ls = styles[label]
        spec = np.abs(np.fft.rfft(results[label]['pax'][:, -1]))
        freqs = np.fft.rfftfreq(nT, dT) / 1e6
        axes[1, 1].plot(freqs, 20 * np.log10(spec / (np.max(spec) + 1e-30) + 1e-30), ls, label=label)
    axes[1, 1].set(xlabel='f (MHz)', ylabel='dB', title='Final waveform spectrum')
    axes[1, 1].set_xlim([0, 4 * f0 / 1e6])
    axes[1, 1].legend();  axes[1, 1].grid(True)

    _save_fig(fig, 'test2_nonlinear.png')
    np.savez(os.path.join(OUTDIR, 'test2_nonlinear.npz'),
             **{f'{k}_{kk}': vv for k, v in results.items() for kk, vv in v.items()})


# ======================================================================
# Test 3 — TVD limiter effect on total variation
# ======================================================================
def test_tvd_limiter():
    print('\n=== Test 3: TVD limiter — total variation ===')
    f0 = 3e6;  c0 = 1500;  lam = c0 / f0
    dX = lam / 5;  dT = dX / (5 * c0)
    # Use higher pressure & coarser step to exercise nonlinear TVD regime
    wX = 5e-3;  wY = 5e-3;  p0 = 0.1e6
    nX = 51;  nY = 51;  nT = 101

    apa, xaxis, yaxis, t = _make_ic(nX, nY, nT, dX, dT, f0, c0, p0, wX, wY)

    results = {}
    for label, use_tvd in [('no_tvd', False), ('tvd', True)]:
        params = SolverParams(
            dX=dX, dY=dX, dT=dT, c0=c0, rho0=1000, beta=3.5,
            alpha0=0.5, attenPow=1, f0=f0, propDist=3e-2,
            boundaryFactor=0.15, useSplitStep=True,
            useAdaptiveFiltering=False,  # k-filtering OFF for both
            useTVD=use_tvd,
            adaptiveFilterStrength=0.7,
            stabilityThreshold=0.2, stabilityRecoveryFactor=0.15,
            dZmin=lam)  # coarse step = 1 wavelength
        field, pnp, ppp, pI, pIloss, zaxis, pax = angular_spectrum_solve(apa, params, verbose=False)
        tv = _total_variation(field)
        results[label] = dict(field=field, tv=tv, zaxis=zaxis, pax=pax)
        print(f'  {label}: TV = {tv:.4e}')

    tv_reduction = (results['no_tvd']['tv'] - results['tvd']['tv']) / results['no_tvd']['tv'] * 100
    print(f'  TV reduction with TVD: {tv_reduction:.1f}%')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, ls in [('no_tvd', '-'), ('tvd', '--')]:
        axes[0].plot(t * 1e6, results[label]['pax'][:, -1], ls, label=label)
    axes[0].set(xlabel='t (us)', ylabel='Pa', title='Final waveform: TVD effect (coarse step)')
    axes[0].legend();  axes[0].grid(True)

    axes[1].bar(['No TVD', 'With TVD'],
                [results['no_tvd']['tv'], results['tvd']['tv']])
    axes[1].set(ylabel='Total Variation', title=f'TV reduction: {tv_reduction:.1f}%')

    _save_fig(fig, 'test3_tvd.png')
    np.savez(os.path.join(OUTDIR, 'test3_tvd.npz'),
             tv_no_tvd=results['no_tvd']['tv'], tv_tvd=results['tvd']['tv'],
             tv_reduction=tv_reduction)
    return tv_reduction


# ======================================================================
# Test 4 — Adaptive frequency filtering
# ======================================================================
def test_adaptive_filtering():
    print('\n=== Test 4: Adaptive frequency filtering ===')
    f0 = 3e6;  c0 = 1500;  lam = c0 / f0
    dX = lam / 5;  dT = dX / (5 * c0)
    nX = 51;  nY = 51;  nT = 101
    # Use large dZ so normalized step is ~1 and cutoff is substantial
    dZ = lam * 5

    HH_nofilt, _ = precalculate_mas(nX, nY, nT, dX, dX, dZ, dT, c0,
                                     adaptive_filtering=False)
    HH_filt, _ = precalculate_mas(nX, nY, nT, dX, dX, dZ, dT, c0,
                                   adaptive_filtering=True,
                                   filter_threshold=0.05,
                                   filter_strength=0.7)

    # Sum |H| over all positive temporal frequencies to see total suppression
    total_nofilt = np.sum(np.abs(HH_nofilt))
    total_filt = np.sum(np.abs(HH_filt))
    suppression = total_nofilt - total_filt
    rel_suppression = suppression / (total_nofilt + 1e-30) * 100

    # Pick a temporal frequency slice near center of passband for visualization
    m = 3 * nT // 4
    H_nf = np.abs(HH_nofilt[:, :, m])
    H_f = np.abs(HH_filt[:, :, m])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].imshow(H_nf, aspect='equal')
    axes[0].set_title('|H| no filtering')
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(H_f, aspect='equal')
    axes[1].set_title('|H| with adaptive filtering')
    plt.colorbar(im1, ax=axes[1])
    diff = H_nf - H_f
    im2 = axes[2].imshow(diff, aspect='equal')
    axes[2].set_title(f'Difference (suppression {rel_suppression:.1f}%)')
    plt.colorbar(im2, ax=axes[2])

    _save_fig(fig, 'test4_adaptive_filter.png')
    print(f'  total suppression = {suppression:.4e} ({rel_suppression:.1f}%)')
    np.savez(os.path.join(OUTDIR, 'test4_filter.npz'),
             H_nf=H_nf, H_f=H_f, suppression=suppression,
             rel_suppression=rel_suppression)
    return rel_suppression


# ======================================================================
# Test 5 — Energy conservation & attenuation loss tracking
# ======================================================================
def test_energy_tracking():
    print('\n=== Test 5: Energy conservation & loss tracking ===')
    f0 = 3e6;  c0 = 1500;  lam = c0 / f0
    dX = lam / 5;  dT = dX / (5 * c0)
    wX = 5e-3;  wY = 5e-3;  p0 = 1e3
    nX = 51;  nY = 51;  nT = 101

    apa, xaxis, yaxis, t = _make_ic(nX, nY, nT, dX, dT, f0, c0, p0, wX, wY)

    # With attenuation
    params = SolverParams(
        dX=dX, dY=dX, dT=dT, c0=c0, rho0=1000, beta=0.0,
        alpha0=0.5, attenPow=1, f0=f0, propDist=2e-2,
        boundaryFactor=0.15, useSplitStep=True,
        useAdaptiveFiltering=False,
        stabilityThreshold=0.2, stabilityRecoveryFactor=0.15,
        dZmin=dX * 4)
    _, _, _, pI, pIloss, zaxis, _ = angular_spectrum_solve(apa, params, verbose=False)

    total_I = np.sum(pI, axis=(0, 1))
    total_loss = np.sum(pIloss, axis=(0, 1))
    cumulative_loss = np.cumsum(total_loss)

    # Without attenuation
    params_noatten = SolverParams(
        dX=dX, dY=dX, dT=dT, c0=c0, rho0=1000, beta=0.0,
        alpha0=-1, f0=f0, propDist=2e-2,
        boundaryFactor=0.15, useSplitStep=True,
        useAdaptiveFiltering=False,
        stabilityThreshold=0.2, stabilityRecoveryFactor=0.15,
        dZmin=dX * 4)
    _, _, _, pI_noatten, pIloss_noatten, zaxis_na, _ = angular_spectrum_solve(apa, params_noatten, verbose=False)
    total_I_na = np.sum(pI_noatten, axis=(0, 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(zaxis * 1e3, total_I, label='With attenuation')
    axes[0].plot(zaxis_na * 1e3, total_I_na, '--', label='No attenuation')
    axes[0].set(xlabel='z (mm)', ylabel='Total intensity', title='Energy vs propagation distance')
    axes[0].legend();  axes[0].grid(True)

    axes[1].plot(zaxis * 1e3, cumulative_loss, label='Cumulative loss (tracked)')
    initial_I = total_I[0] if len(total_I) > 0 else 0
    axes[1].plot(zaxis * 1e3, initial_I - total_I, '--', label='Actual I drop')
    axes[1].set(xlabel='z (mm)', ylabel='Intensity loss', title='Attenuation loss tracking')
    axes[1].legend();  axes[1].grid(True)

    _save_fig(fig, 'test5_energy.png')
    np.savez(os.path.join(OUTDIR, 'test5_energy.npz'),
             total_I=total_I, cumulative_loss=cumulative_loss, zaxis=zaxis)


# ======================================================================
# Test 6 — Convergence: sequential vs split-step
# ======================================================================
def test_convergence():
    print('\n=== Test 6: Convergence — sequential vs split-step vs KT ===')
    f0 = 3e6;  c0 = 1500;  lam = c0 / f0
    dX = lam / 5;  dT = dX / (5 * c0)
    # Strongly nonlinear pre-shock regime: at p0 = 3 MPa, beta = 3.5,
    # f0 = 3 MHz the shock-formation distance is z_shock ~ 17 mm; the
    # 15-mm propagation path reaches ~88% of z_shock so the waveform is
    # visibly steepened and the Strang+KT splitting error dominates
    # over the fp32 / attenuation-filter floor that caps the slope in
    # weak-nonlinearity regimes.
    wX = 5e-3;  wY = 5e-3;  p0 = 3.0e6
    nX = 51;  nY = 51;  nT = 401

    apa, xaxis, yaxis, t = _make_ic(nX, nY, nT, dX, dT, f0, c0, p0, wX, wY)

    # dZ range chosen so the splitting error at the coarsest point is
    # well above the fp32 floor; pairwise (Cauchy) differencing then
    # reveals the observed order of convergence.  split_kt's accuracy
    # at dZ = 16 dX is already ~1e-4 relative, so coarser steps are
    # needed to see the quadratic decrease cleanly.
    dZ_values = [dX * 32, dX * 16, dX * 8, dX * 4, dX * 2]

    # Run sequential, split-step (Rusanov), and split-step (KT) at each resolution
    configs = [
        ('sequential',     dict(useSplitStep=False, fluxScheme='rusanov')),
        ('split_rusanov',  dict(useSplitStep=True,  fluxScheme='rusanov')),
        ('split_kt',       dict(useSplitStep=True,  fluxScheme='kt')),
    ]
    all_results = {}
    for label, kw in configs:
        results = {}
        for dZ_val in dZ_values:
            params = SolverParams(
                dX=dX, dY=dX, dT=dT, c0=c0, rho0=1000, beta=3.5,
                alpha0=0.1, attenPow=1, f0=f0, propDist=1.5e-2,
                boundaryFactor=0.15, useBoundaryLayer=False,
                useAdaptiveFiltering=False, useTVD=False,
                stabilityThreshold=1e9, stabilityRecoveryFactor=0.15,
                dZmin=dZ_val, **kw)
            field, _, _, pI, _, zaxis, pax = angular_spectrum_solve(apa, params, verbose=False)
            results[dZ_val] = dict(
                field=field,
                pI=pI,
                zaxis=zaxis,
                pax=pax,
                center_trace=field[nX // 2, nY // 2, :].copy(),
            )
        all_results[label] = results
        print(f'  {label}: done')

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Pairwise (Cauchy) convergence: compare each dZ to its dZ/2 counterpart.
    # This avoids depending on a single fine reference whose own fp32 noise
    # accumulates over many small steps and saturates the error floor.  For
    # a scheme with global order p, the pair difference scales as dZ^p; the
    # log-log slope of pair-error vs dZ recovers p directly.
    all_errors = {}
    dZ_pairs = dZ_values[:-1]   # coarser of each pair; paired with next (dZ/2)
    for label, _ in configs:
        errors = []
        for i, dZ_val in enumerate(dZ_pairs):
            trace_coarse = all_results[label][dZ_val]['center_trace']
            trace_fine   = all_results[label][dZ_values[i+1]]['center_trace']
            ref_norm = np.sqrt(np.mean(trace_fine ** 2)) + 1e-30
            err = np.sqrt(np.mean((trace_coarse - trace_fine) ** 2)) / ref_norm
            errors.append(err)
        all_errors[label] = errors

    # Panel 1: waveforms at coarsest step (anchor trace at finest tested dZ).
    dZ_coarse = dZ_values[0]
    dZ_fine   = dZ_values[-1]
    for label, _ in configs:
        axes[0].plot(t * 1e6, all_results[label][dZ_coarse]['center_trace'], label=label)
    anchor_label = 'split_kt'
    axes[0].plot(t * 1e6, all_results[anchor_label][dZ_fine]['center_trace'],
                 'k:', lw=0.8, label=f'{anchor_label} at dZ/dX={dZ_fine/dX:.1f}')
    axes[0].set(xlabel='t (us)', ylabel='Pa', title=f'Waveform at dZ/dX={dZ_coarse/dX:.0f}')
    axes[0].legend(fontsize=8);  axes[0].grid(True)

    # Panel 2: pairwise convergence curves.
    dZ_norm = np.array([d / dX for d in dZ_pairs])
    markers = ['s', 'o', '^']
    for (label, _), marker in zip(configs, markers):
        axes[1].loglog(dZ_norm, all_errors[label], f'{marker}-', label=label)

    dz_ref = np.array([dZ_norm[0], dZ_norm[-1]])
    e0 = all_errors['split_kt'][0]
    axes[1].loglog(dz_ref, e0 * (dz_ref / dz_ref[0])**1, 'k--', lw=0.7, alpha=0.5, label='O(dz)')
    axes[1].loglog(dz_ref, e0 * (dz_ref / dz_ref[0])**2, 'k:', lw=0.7, alpha=0.5, label='O(dz$^2$)')
    axes[1].set(xlabel='dZ / dX',
                ylabel='Pairwise L2 error (dZ vs dZ/2)',
                title='Convergence rate comparison')
    axes[1].legend(fontsize=7);  axes[1].grid(True, which='both')

    plt.tight_layout()
    _save_fig(fig, 'test6_convergence.png')

    # Convergence rate: log-linear fit across the coarsest three pair-points,
    # where the splitting error dominates over any solver floor.
    rates = {}
    for label, _ in configs:
        errs = np.array(all_errors[label])
        valid = np.isfinite(errs) & (errs > 1e-15)
        if valid.sum() >= 3:
            x = np.log(np.array(dZ_pairs)[valid][:3])
            y = np.log(errs[valid][:3])
            rate = np.polyfit(x, y, 1)[0]
            rates[label] = rate
            print(f'  {label:20s} convergence rate: {rate:.2f}')

    np.savez(os.path.join(OUTDIR, 'test6_convergence.npz'),
             dZ_values=dZ_values,
             dZ_pairs=dZ_pairs,
             **{f'errors_{l}': all_errors[l] for l, _ in configs},
             rates=rates)
    return all_errors, rates


# ======================================================================
# Test 7 — Isolated nonlinear-operator order
# ======================================================================
def _run_nonlinear_only(field0, step_fn, N, dZ, dT, z_final):
    """Advance the nonlinear operator alone for a fixed propagation distance."""
    n_steps = int(round(z_final / dZ))
    field = field0.copy()
    for _ in range(n_steps):
        field = np.asarray(step_fn(field, N, dZ, dT))
    return field


def test_nonlinear_operator_order():
    print('\n=== Test 7: Isolated nonlinear-operator order ===')

    # Smooth pulse chosen to avoid immediate shock formation while still
    # making nonlinear error measurable on coarse steps.
    nT = 801
    t = np.linspace(-1.0, 1.0, nT)
    dT = t[1] - t[0]
    amp = 1.5
    z_final = 0.03
    N = 1.0

    u0 = amp * np.exp(-(t / 0.22) ** 2) * np.sin(2 * np.pi * 4 * t)
    u0[[0, -1]] = 0.0
    field0 = u0.reshape(1, 1, -1).astype(np.float32)

    dZ_values = [z_final / 32, z_final / 64, z_final / 128, z_final / 256, z_final / 512]
    ref_dZ = z_final / 4096
    schemes = {
        'Rusanov': _rusanov_flux_standard,
        'KT': _kt_flux,
    }

    results = {}
    for name, step_fn in schemes.items():
        ref = _run_nonlinear_only(field0, step_fn, N, ref_dZ, dT, z_final)[0, 0]
        errors = []
        finals = {}
        for dZ in dZ_values:
            u = _run_nonlinear_only(field0, step_fn, N, dZ, dT, z_final)[0, 0]
            err = np.sqrt(np.mean((u - ref) ** 2))
            errors.append(err)
            finals[dZ] = u

        # Fit only the coarsest three points to avoid the fine-grid error floor.
        fit_x = np.log(np.array(dZ_values[:3]))
        fit_y = np.log(np.array(errors[:3]))
        rate = np.polyfit(fit_x, fit_y, 1)[0]
        results[name] = dict(ref=ref, errors=np.array(errors), finals=finals, rate=rate)
        print(f'  {name:8s} coarse-grid order estimate: {rate:.2f}')
        print('    errors:', [f'{e:.3e}' for e in errors])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    dz_norm = np.array(dZ_values) / dT
    axes[0].loglog(dz_norm, results['Rusanov']['errors'], 'o-', label=f"Rusanov (p≈{results['Rusanov']['rate']:.2f})")
    axes[0].loglog(dz_norm, results['KT']['errors'], 's-', label=f"KT (p≈{results['KT']['rate']:.2f})")
    e0 = results['Rusanov']['errors'][0]
    dz_ref = np.array([dz_norm[0], dz_norm[2]])
    axes[0].loglog(dz_ref, e0 * (dz_ref / dz_ref[0]) ** 1, 'k--', lw=0.8, alpha=0.6, label='O(dz)')
    axes[0].loglog(dz_ref, e0 * (dz_ref / dz_ref[0]) ** 2, 'k:', lw=0.8, alpha=0.6, label='O(dz$^2$)')
    axes[0].set(xlabel='dZ / dT', ylabel='L2 error vs fine reference', title='Isolated nonlinear-step convergence')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which='both')

    dZ_show = dZ_values[0]
    axes[1].plot(t, field0[0, 0], 'k:', lw=1.0, label='Initial')
    axes[1].plot(t, results['Rusanov']['ref'], 'k-', lw=1.5, label=f'Reference (dZ={ref_dZ:.2e})')
    axes[1].plot(t, results['Rusanov']['finals'][dZ_show], '--', label=f'Rusanov (dZ={dZ_show:.2e})')
    axes[1].plot(t, results['KT']['finals'][dZ_show], '-.', label=f'KT (dZ={dZ_show:.2e})')
    axes[1].set(xlabel='t', ylabel='u', title='Final waveform after nonlinear-only evolution')
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    plt.tight_layout()
    _save_fig(fig, 'test7_nonlinear_operator_order.png')
    np.savez(
        os.path.join(OUTDIR, 'test7_nonlinear_operator_order.npz'),
        dZ_values=np.array(dZ_values),
        ref_dZ=ref_dZ,
        errors_rusanov=results['Rusanov']['errors'],
        errors_kt=results['KT']['errors'],
        rate_rusanov=results['Rusanov']['rate'],
        rate_kt=results['KT']['rate'],
    )
    return results


# ======================================================================
# Test 8 — TVD validity and stability in the shock regime
# ======================================================================
def test_tvd_shock_step_size():
    print('\n=== Test 8: TVD shock-regime validity and stability vs step size ===')

    nT = 801
    t = np.linspace(-1.0, 1.0, nT)
    dT = t[1] - t[0]
    amp = 2.5
    z_final = 0.05
    N = 1.0
    beta_tvd = 1.0

    # Smooth initial pulse that steepens strongly under Burgers evolution.
    u0 = amp * np.exp(-(t / 0.18) ** 2) * np.sin(2 * np.pi * 4 * t)
    u0[[0, -1]] = 0.0
    field0 = u0.reshape(1, 1, -1).astype(np.float32)

    dZ_values = [z_final / 16, z_final / 32, z_final / 64, z_final / 128, z_final / 256]
    ref_dZ = z_final / 4096

    def _step_std(field, N, dZ, dT):
        return _rusanov_flux_standard(field, N, dZ, dT)

    def _step_tvd(field, N, dZ, dT):
        return _rusanov_flux_tvd(field, N, dZ, dT, beta_tvd)

    def _step_kt(field, N, dZ, dT):
        return _kt_flux(field, N, dZ, dT)

    def _step_kt_mm(field, N, dZ, dT):
        return _kt_flux_minmod(field, N, dZ, dT)

    def _step_kt_adaptive(field, N, dZ, dT):
        return _kt_flux_adaptive(field, N, dZ, dT)

    schemes = {
        'Rusanov': _step_std,
        'KT': _step_kt_mm,
        'KT-adaptive': _step_kt_adaptive,
    }

    ref = _run_nonlinear_only(field0, _step_kt_adaptive, N, ref_dZ, dT, z_final)[0, 0]
    results = {}
    for name, step_fn in schemes.items():
        errors = []
        stable = []
        finals = {}
        tvs = []
        peaks = []
        troughs = []
        for dZ in dZ_values:
            u = _run_nonlinear_only(field0, step_fn, N, dZ, dT, z_final)[0, 0]
            finals[dZ] = u
            is_stable = np.isfinite(u).all()
            stable.append(is_stable)
            if is_stable:
                err = np.sqrt(np.mean((u - ref) ** 2))
                tv = np.sum(np.abs(np.diff(u)))
                peak = np.max(u)
                trough = np.min(u)
            else:
                err = np.nan
                tv = np.nan
                peak = np.nan
                trough = np.nan
            errors.append(err)
            tvs.append(tv)
            peaks.append(peak)
            troughs.append(trough)
            print(f'  {name:7s} dZ={dZ:.3e}: stable={is_stable}, err={err}, TV={tv}')

        results[name] = dict(
            errors=np.array(errors, dtype=np.float64),
            stable=np.array(stable, dtype=bool),
            finals=finals,
            tvs=np.array(tvs, dtype=np.float64),
            peaks=np.array(peaks, dtype=np.float64),
            troughs=np.array(troughs, dtype=np.float64),
        )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    scheme_styles = [('Rusanov', 'o'), ('KT', '^'), ('KT-adaptive', 'D')]

    # Panel 1: validity vs step size using error against fine KT reference.
    for name, marker in scheme_styles:
        mask = results[name]['stable']
        dz_plot = np.array(dZ_values)[mask]
        err_plot = results[name]['errors'][mask]
        if len(dz_plot):
            axes[0].loglog(dz_plot, err_plot, f'{marker}-', label=name)
        dz_bad = np.array(dZ_values)[~mask]
        if len(dz_bad):
            err_floor = np.nanmin([r['errors'][r['stable']].min()
                                   for r in results.values() if np.any(r['stable'])]) * 2
            axes[0].plot(dz_bad, np.full_like(dz_bad, err_floor),
                         marker, linestyle='None', markersize=8,
                         markerfacecolor='none', label=f'{name} unstable')
    axes[0].set(xlabel='dZ', ylabel='L2 error vs fine KT reference',
                title='Validity vs step size')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which='both')

    # Panel 2: total variation as a stability proxy.
    for name, marker in scheme_styles:
        mask = results[name]['stable']
        if np.any(mask):
            axes[1].plot(np.array(dZ_values)[mask], results[name]['tvs'][mask],
                         f'{marker}-', label=name)
    axes[1].set(xlabel='dZ', ylabel='Total variation',
                title='Shock-regime TV vs step size')
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    # Panel 3: coarsest stable waveforms.
    dZ_show = dZ_values[0]
    axes[2].plot(t, u0, 'k:', lw=1.0, label='Initial')
    axes[2].plot(t, ref, 'k-', lw=1.4, label=f'Reference (dZ={ref_dZ:.2e})')
    for name, ls in [('Rusanov', '--'), ('KT', '-'), ('KT-adaptive', '-')]:
        u = results[name]['finals'][dZ_show]
        if np.isfinite(u).all():
            axes[2].plot(t, u, ls, label=f'{name} (dZ={dZ_show:.2e})')
    # Set y-axis to the reference range
    ymax = 1.3 * max(np.max(np.abs(ref)), np.max(np.abs(u0)))
    axes[2].set(xlabel='t', ylabel='u', title='Coarsest-step final waveform',
                ylim=[-ymax, ymax])
    axes[2].legend(fontsize=7)
    axes[2].grid(True)

    plt.tight_layout()
    _save_fig(fig, 'test8_tvd_shock_step_size.png')
    np.savez(
        os.path.join(OUTDIR, 'test8_tvd_shock_step_size.npz'),
        dZ_values=np.array(dZ_values),
        ref_dZ=ref_dZ,
        **{f'errors_{n}': results[n]['errors'] for n in results},
        **{f'stable_{n}': results[n]['stable'] for n in results},
        **{f'tv_{n}': results[n]['tvs'] for n in results},
    )
    return results


# ======================================================================
# Test 9 — Synthetic split linear+nonlinear order
# ======================================================================
def test_synthetic_split_kt_order():
    print('\n=== Test 9: Synthetic split linear+nonlinear order (Strang + KT) ===')

    nT = 801
    t = np.linspace(-1.0, 1.0, nT)
    dT = t[1] - t[0]
    z_final = 0.05
    N = 1.0

    # Smooth waveform and an exact linear dispersive-diffusive operator.
    amp = 1.5
    kappa = 0.02
    gamma = 0.002
    u0 = amp * np.exp(-(t / 0.22) ** 2) * np.sin(2 * np.pi * 4 * t)
    u0[[0, -1]] = 0.0
    field0 = u0.reshape(1, 1, -1).astype(np.float32)

    freqs = np.fft.rfftfreq(nT, dT)
    omega = 2 * np.pi * freqs

    def _linear_half_step(field, dZ):
        filt_half = np.exp((-gamma * omega ** 2 - 1j * kappa * omega ** 2) * dZ / 2)
        filt_half = jnp.array(filt_half.reshape(1, 1, -1), dtype=jnp.complex64)
        return np.asarray(
            jnp.fft.irfft(jnp.fft.rfft(field, axis=2) * filt_half, n=nT, axis=2)
        )

    def _split_kt_step(field, dZ):
        field = _linear_half_step(field, dZ)
        field = np.asarray(_kt_flux(jnp.array(field), N, dZ, dT))
        field = _linear_half_step(field, dZ)
        return field

    def _run(field, dZ):
        n_steps = int(round(z_final / dZ))
        out = field.copy()
        for _ in range(n_steps):
            out = _split_kt_step(out, dZ)
        return out

    dZ_values = [z_final / 16, z_final / 32, z_final / 64, z_final / 128]
    ref_dZ = z_final / 4096

    ref = _run(field0, ref_dZ)[0, 0]
    errors = []
    finals = {}
    for dZ in dZ_values:
        u = _run(field0, dZ)[0, 0]
        finals[dZ] = u
        err = np.sqrt(np.mean((u - ref) ** 2))
        errors.append(err)
        print(f'  dZ={dZ:.3e}: error={err:.3e}')

    # Fit the coarsest three points before the fine-grid error floor appears.
    fit_x = np.log(np.array(dZ_values[:3]))
    fit_y = np.log(np.array(errors[:3]))
    rate = np.polyfit(fit_x, fit_y, 1)[0]
    print(f'  coarse-grid order estimate: {rate:.2f}')

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))

    axes[0].loglog(np.array(dZ_values) / z_final, errors, 's-', label=f'KT split-step (p≈{rate:.2f})')
    e0 = errors[0]
    dz_ref = np.array([dZ_values[0], dZ_values[2]]) / z_final
    axes[0].loglog(dz_ref, e0 * (dz_ref / dz_ref[0]) ** 1, 'k--', lw=0.8, alpha=0.6, label='O(dz)')
    axes[0].loglog(dz_ref, e0 * (dz_ref / dz_ref[0]) ** 2, 'k:', lw=0.8, alpha=0.6, label='O(dz$^2$)')
    axes[0].set(xlabel='dZ / z_final', ylabel='L2 error vs fine reference',
                title='Synthetic split linear+nonlinear convergence')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which='both')

    dZ_show = dZ_values[0]
    axes[1].plot(t, u0, 'k:', lw=1.0, label='Initial')
    axes[1].plot(t, ref, 'k-', lw=1.5, label=f'Reference (dZ={ref_dZ:.2e})')
    axes[1].plot(t, finals[dZ_show], '--', label=f'KT (dZ={dZ_show:.2e})')
    axes[1].set(xlabel='t', ylabel='u', title='Coarsest-step synthetic split solution')
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    plt.tight_layout()
    _save_fig(fig, 'test9_synthetic_split_kt_order.png')
    np.savez(
        os.path.join(OUTDIR, 'test9_synthetic_split_kt_order.npz'),
        dZ_values=np.array(dZ_values),
        ref_dZ=ref_dZ,
        errors=np.array(errors),
        rate=rate,
    )
    return dict(errors=np.array(errors), rate=rate, ref_dZ=ref_dZ)


# ======================================================================
# Test 10 — Attenuation & Kramers-Kronig dispersion validation
# ======================================================================
def test_attenuation_dispersion():
    """Validate attenuation and KK dispersion against theory.

    Propagates a broadband pulse through the solver and measures the
    per-frequency attenuation and phase velocity from the input/output
    spectra.  Compares against the theoretical power-law attenuation
    and Kramers-Kronig phase-velocity dispersion over the soft-tissue
    range 0.5–10 MHz.
    """
    print('\n=== Test 10: Attenuation & Kramers-Kronig dispersion ===')

    # --- Parameters ---
    f0 = 3e6;  c0 = 1540.0;  lam = c0 / f0
    alpha0 = 0.5   # dB/cm/MHz — typical soft tissue
    pw = 1          # linear-in-f power law
    rho0 = 1000.0

    # Grid — fine temporal sampling for broadband content up to 15 MHz
    # Spatial grid dX controls lateral beam resolution, not temporal bandwidth.
    # For this plane-wave test (uniform nX×nY aperture), only the temporal
    # sampling dT limits the measurable frequency range.
    dX = lam / 3
    dT = 1.0 / (30e6 * 2)  # Nyquist at 30 MHz
    nX = 5;  nY = 5         # minimal lateral (plane-wave-like)
    nT = 1001               # long time window
    propDist = 20e-3         # 2 cm propagation

    t = np.arange(nT) * dT
    t -= np.mean(t)

    # Broadband IC: very short Gaussian pulse for flat spectrum to 10+ MHz
    sigma_t = 0.15 / f0   # ~50 ns — bandwidth covers well past 10 MHz
    ic_vec = np.exp(-0.5 * (t / sigma_t) ** 2) * np.sin(2 * np.pi * f0 * t)
    apa = np.ones((nX, nY, 1)) * ic_vec[np.newaxis, np.newaxis, :]
    apa = apa.astype(np.float32)

    # --- Run solver ---
    params = SolverParams(
        dX=dX, dY=dX, dT=dT, c0=c0, rho0=rho0, beta=0.0,
        alpha0=alpha0, attenPow=pw, f0=f0, propDist=propDist,
        boundaryFactor=0.0, useBoundaryLayer=False,
        useSplitStep=True, fluxScheme='kt',
        useAdaptiveFiltering=False, useTVD=False,
        stabilityThreshold=0.2, stabilityRecoveryFactor=0.15,
        dZmin=dX)
    field, _, _, _, _, zaxis, _ = angular_spectrum_solve(apa, params, verbose=False)

    z_prop = zaxis[-1]
    ic_center = apa[nX // 2, nY // 2, :]
    out_center = np.array(field[nX // 2, nY // 2, :])

    # --- Spectral analysis ---
    F_in = np.fft.rfft(ic_center)
    F_out = np.fft.rfft(out_center)
    freqs = np.fft.rfftfreq(nT, dT)

    # Avoid divide-by-zero: only analyse bins with significant input energy
    mag_in = np.abs(F_in)
    threshold = np.max(mag_in) * 1e-3
    valid = mag_in > threshold

    # Measured attenuation: |F_out/F_in| = exp(-alpha * z)
    ratio = np.abs(F_out[valid]) / mag_in[valid]
    alpha_measured_Np = -np.log(ratio + 1e-30) / z_prop   # Np/m
    alpha_measured_dB = alpha_measured_Np * 20 * np.log10(np.e) / 1e2  # dB/cm

    # Measured phase velocity from phase difference
    phase_diff = np.angle(F_out[valid] / F_in[valid])
    # Unwrap to remove 2π jumps
    phase_diff = np.unwrap(phase_diff)
    # Phase advance = omega * z / c_phase  (relative to c0 propagation already done by AS)
    # The AS propagates at c0; the attenuation filter adds the KK correction.
    # Total accumulated phase = phase_diff = -alphaStar * z
    # Phase velocity: 1/c_p = 1/c0 + alphaStar/(2*pi*f)
    omega_valid = 2 * np.pi * freqs[valid]
    f_valid = freqs[valid]

    # --- Theoretical curves ---
    conv = alpha0 / (1e6 ** pw) * 1e2 / (20 * np.log10(np.e))
    f_theory = np.linspace(0.5e6, 15e6, 500)
    alpha_theory_Np = conv * f_theory ** pw
    alpha_theory_dB = alpha0 * (f_theory / 1e6) ** pw  # dB/cm/MHz^pw * (f/MHz)^pw

    # KK phase velocity
    if pw % 2 == 1:
        f_theory_safe = np.maximum(f_theory, 1.0)
        alphaStar0_theory = (-2 * conv / ((2 * np.pi) ** pw) / np.pi) * \
                            (np.log(2 * np.pi * f_theory_safe) - np.log(2 * np.pi * f0))
    else:
        alphaStar0_theory = (conv / (2 * np.pi) ** pw) * np.tan(np.pi * pw / 2) * \
                            ((2 * np.pi * f_theory) ** (pw - 1) - (2 * np.pi * f0) ** (pw - 1))
    alphaStar_theory = 2 * np.pi * alphaStar0_theory * f_theory
    c_theory = 1.0 / (1.0 / c0 + alphaStar_theory / (2 * np.pi * f_theory))

    # Measured phase velocity from solver
    # alphaStar_meas = -phase_diff / z_prop (from the attenuation filter's contribution)
    # But the total phase_diff includes the AS propagation. Since AS uses c0,
    # and the total field accumulates (k-k0)*z from dispersion:
    # phase_diff_disp = phase_diff  (AS already removed the k0*z baseline)
    alphaStar_meas = -phase_diff / z_prop
    c_meas = 1.0 / (1.0 / c0 + alphaStar_meas / (omega_valid + 1e-30))

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Attenuation vs frequency (dB/cm)
    axes[0, 0].plot(f_valid / 1e6, alpha_measured_dB, 'bo', ms=3, alpha=0.6,
                     label='Measured (solver)')
    axes[0, 0].plot(f_theory / 1e6, alpha_theory_dB, 'r-', lw=2,
                     label=f'Theory: {alpha0} dB/cm/MHz')
    axes[0, 0].set(xlabel='Frequency (MHz)', ylabel='Attenuation (dB/cm)',
                    title='Attenuation vs frequency',
                    xlim=[0.5, 10], ylim=[0, 6])
    axes[0, 0].legend();  axes[0, 0].grid(True)

    # Panel 2: Attenuation error
    # Interpolate theory to measured frequencies
    alpha_theory_at_meas = alpha0 * (f_valid / 1e6) ** pw
    alpha_err = alpha_measured_dB - alpha_theory_at_meas
    axes[0, 1].plot(f_valid / 1e6, alpha_err, 'b-', lw=1)
    axes[0, 1].axhline(0, color='k', ls=':', lw=0.5)
    axes[0, 1].set(xlabel='Frequency (MHz)', ylabel='Error (dB/cm)',
                    title='Attenuation error (measured - theory)',
                    xlim=[0.5, 10], ylim=[-0.005, 0.005])
    axes[0, 1].grid(True)

    # Panel 3: Phase velocity vs frequency
    axes[1, 0].plot(f_valid / 1e6, c_meas, 'bo', ms=3, alpha=0.6,
                     label='Measured (solver)')
    axes[1, 0].plot(f_theory / 1e6, c_theory, 'r-', lw=2,
                     label='Kramers-Kronig theory')
    axes[1, 0].axhline(c0, color='k', ls=':', lw=0.5, label=f'c₀ = {c0:.0f} m/s')
    axes[1, 0].set(xlabel='Frequency (MHz)', ylabel='Phase velocity (m/s)',
                    title='Phase velocity (Kramers-Kronig dispersion)',
                    xlim=[0.5, 10])
    axes[1, 0].legend();  axes[1, 0].grid(True)

    # Panel 4: Waveforms
    axes[1, 1].plot(t * 1e6, ic_center, 'b-', lw=1, label='Input')
    axes[1, 1].plot(t * 1e6, out_center, 'r-', lw=1,
                     label=f'Output (z={z_prop*1e3:.0f} mm)')
    axes[1, 1].set(xlabel='Time (μs)', ylabel='Pressure (Pa)',
                    title='Waveform comparison',
                    xlim=[-1, 1])
    axes[1, 1].legend();  axes[1, 1].grid(True)

    fig.suptitle(f'Attenuation & KK Dispersion: α₀={alpha0} dB/cm/MHz, '
                 f'c₀={c0:.0f} m/s, f₀={f0/1e6:.0f} MHz', fontsize=13)
    plt.tight_layout()
    _save_fig(fig, 'test10_attenuation_kk.png')

    # Compute fit quality in the 1–5 MHz range
    band = (f_valid >= 1e6) & (f_valid <= 5e6)
    if np.any(band):
        atten_rms = np.sqrt(np.mean(alpha_err[band] ** 2))
        print(f'  Attenuation RMS error (1-5 MHz): {atten_rms:.4f} dB/cm')
    else:
        atten_rms = np.nan
        print('  No data in 1-5 MHz band')

    np.savez(os.path.join(OUTDIR, 'test10_attenuation_kk.npz'),
             f_valid=f_valid, alpha_measured_dB=alpha_measured_dB,
             c_meas=c_meas, f_theory=f_theory,
             alpha_theory_dB=alpha_theory_dB, c_theory=c_theory,
             atten_rms=atten_rms)
    return atten_rms


# ======================================================================
# Main
# ======================================================================
if __name__ == '__main__':
    print('Angular Spectrum Solver — Validation Suite')
    print('=' * 60)

    r1 = test_linear_propagation()
    test_nonlinear_comparison()
    r3 = test_tvd_limiter()
    r4 = test_adaptive_filtering()
    test_energy_tracking()
    r6 = test_convergence()
    r7 = test_nonlinear_operator_order()
    r8 = test_tvd_shock_step_size()
    r9 = test_synthetic_split_kt_order()
    r10 = test_attenuation_dispersion()

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'Test 1 — Linear: seq vs split-step relative diff = {r1:.4e}')
    print(f'Test 3 — TVD: total-variation reduction = {r3:.1f}%')
    print(f'Test 4 — Adaptive filter suppression = {r4:.4e}')
    print(f'Test 6 — Convergence errors: {[f"{e:.4e}" for e in r6]}')
    print(f"Test 7 — Nonlinear-only order: Rusanov={r7['Rusanov']['rate']:.2f}, KT={r7['KT']['rate']:.2f}")
    print(f"Test 8 — Shock stability at coarsest step: "
          f"Rusanov={r8['Rusanov']['stable'][0]}, "
          f"KT={r8['KT']['stable'][0]}, "
          f"KT-adaptive={r8['KT-adaptive']['stable'][0]}")
    print(f"Test 9 — Synthetic split KT order: KT={r9['rate']:.2f}")
    print('\nAll results saved to validation_results/')
