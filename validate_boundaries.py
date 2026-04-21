"""
Validation of improved boundary conditions for the angular spectrum solver.

Tests:
  1. Taper profile comparison: quadratic vs Wendland C2
  2. Boundary reflection measurement: propagate a plane wave toward the
     boundary and measure the reflected energy
  3. Frequency-weighted boundary: measure reflection at low vs high freq
  4. Super-absorbing boundary: directional decomposition effectiveness
  5. Combined: all improvements together vs baseline

Results saved to validation_results/boundaries/
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from angular_spectrum_solver import (
    SolverParams, angular_spectrum_solve,
    ablvec, ablvec_wendland, precalculate_abl, precalculate_abl_freq,
    _make_boundary_weights,
)

OUTDIR = os.path.join(os.path.dirname(__file__), 'validation_results', 'boundaries')
os.makedirs(OUTDIR, exist_ok=True)


def _save_fig(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def _make_ic(nX, nY, nT, dX, dT, f0, c0, p0, wX, wY, ncycles=1.5):
    xaxis = np.arange(nX) * dX - (nX - 1) * dX / 2
    yaxis = np.arange(nY) * dX - (nY - 1) * dX / 2
    omega0 = 2 * np.pi * f0
    dur = 2
    t = np.arange(nT) * dT
    t -= np.mean(t)
    X, Y = np.meshgrid(xaxis, yaxis, indexing='ij')
    ap = np.where((np.abs(X) <= wX / 2) & (np.abs(Y) <= wY / 2), 1.0, 0.0)
    icvec = np.exp(-(1.05 * t * omega0 / (ncycles * np.pi)) ** (2 * dur)) * np.sin(t * omega0) * p0
    apa = ap[:, :, np.newaxis] * icvec[np.newaxis, np.newaxis, :]
    return apa, xaxis, yaxis, t


# ======================================================================
# Test 1 — Taper profile shapes
# ======================================================================
def test_profiles():
    print('\n=== Test 1: Taper profile comparison ===')
    N, n = 101, 20
    q = ablvec(N, n)
    w = ablvec_wendland(N, n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(q, label='Quadratic')
    axes[0].plot(w, label='Wendland C2')
    axes[0].set(xlabel='Grid index', ylabel='Transmission',
                title='Boundary taper profiles')
    axes[0].legend(); axes[0].grid(True)

    # Derivatives (smoothness comparison)
    dq = np.gradient(np.gradient(q))  # 2nd derivative
    dw = np.gradient(np.gradient(w))
    axes[1].plot(dq, label='Quadratic d2/dx2')
    axes[1].plot(dw, label='Wendland C2 d2/dx2')
    axes[1].set(xlabel='Grid index', ylabel='d2(transmission)/dx2',
                title='Second derivative (smoothness)')
    axes[1].legend(); axes[1].grid(True)

    _save_fig(fig, 'test1_profiles.png')

    # Quantify discontinuity at taper onset
    q_jump = np.max(np.abs(np.diff(dq[n-3:n+3])))
    w_jump = np.max(np.abs(np.diff(dw[n-3:n+3])))
    print(f'  Quadratic: max d2 jump at onset = {q_jump:.4e}')
    print(f'  Wendland:  max d2 jump at onset = {w_jump:.4e}')
    return q_jump, w_jump


# ======================================================================
# Test 2 — Boundary reflection comparison
# ======================================================================
def test_reflection():
    print('\n=== Test 2: Boundary reflection measurement ===')
    f0 = 3e6; c0 = 1500; lam = c0 / f0
    dX = lam / 5; dT = dX / (5 * c0)
    # Wide aperture to send energy toward boundaries
    wX = 8e-3; wY = 8e-3; p0 = 1e3
    nX = 81; nY = 81; nT = 101

    apa, xaxis, yaxis, t = _make_ic(nX, nY, nT, dX, dT, f0, c0, p0, wX, wY)

    configs = [
        ('quadratic',      dict(boundaryProfile='quadratic',
                                useFreqWeightedBoundary=False,
                                useSuperAbsorbing=False)),
        ('wendland',       dict(boundaryProfile='wendland',
                                useFreqWeightedBoundary=False,
                                useSuperAbsorbing=False)),
        ('freq_weighted',  dict(boundaryProfile='wendland',
                                useFreqWeightedBoundary=True,
                                useSuperAbsorbing=False)),
        ('all_combined',   dict(boundaryProfile='wendland',
                                useFreqWeightedBoundary=True,
                                useSuperAbsorbing=True,
                                superAbsorbingStrength=0.5)),
    ]

    results = {}
    for label, kw in configs:
        params = SolverParams(
            dX=dX, dY=dX, dT=dT, c0=c0, rho0=1000, beta=0.0,
            alpha0=-1, f0=f0, propDist=3e-2, boundaryFactor=0.15,
            useSplitStep=True, useAdaptiveFiltering=False, useTVD=False,
            dZmin=dX * 4, **kw)
        field, pnp, ppp, pI, pIloss, zaxis, pax = angular_spectrum_solve(
            apa, params, verbose=False)

        # Measure reflected energy: energy near boundaries vs center
        bdy_width = int(nX * 0.15)
        center = slice(bdy_width, nX - bdy_width)
        I_total = np.sum(pI[:, :, -1])
        I_boundary = np.sum(pI[:bdy_width, :, -1]) + np.sum(pI[-bdy_width:, :, -1]) + \
                     np.sum(pI[center, :bdy_width, -1]) + np.sum(pI[center, -bdy_width:, -1])
        I_center = np.sum(pI[center, center, -1])
        reflection_ratio = I_boundary / (I_center + 1e-30) if I_center > 0 else 0

        results[label] = dict(pI=pI, zaxis=zaxis, field=field,
                              reflection_ratio=reflection_ratio)
        print(f'  {label:20s}: boundary/center ratio = {reflection_ratio:.4e}')

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, (label, r) in enumerate(results.items()):
        ax = axes[idx // 2, idx % 2]
        im = ax.imshow(r['pI'][:, nY // 2, :], aspect='auto',
                       extent=[r['zaxis'][0]*1e3, r['zaxis'][-1]*1e3,
                               xaxis[0]*1e3, xaxis[-1]*1e3])
        ax.set(xlabel='z (mm)', ylabel='x (mm)',
               title=f'{label}\nrefl ratio={r["reflection_ratio"]:.2e}')
        plt.colorbar(im, ax=ax)

    fig.suptitle('Boundary reflection: x-z intensity (central y-slice)', fontsize=14)
    plt.tight_layout()
    _save_fig(fig, 'test2_reflection.png')

    # Bar chart
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    labels = list(results.keys())
    ratios = [results[l]['reflection_ratio'] for l in labels]
    ax2.bar(labels, ratios)
    ax2.set(ylabel='Boundary/Center Intensity Ratio',
            title='Boundary Reflection Comparison')
    ax2.tick_params(axis='x', rotation=15)
    _save_fig(fig2, 'test2_reflection_bar.png')

    np.savez(os.path.join(OUTDIR, 'test2_reflection.npz'),
             labels=labels, ratios=ratios)
    return results


# ======================================================================
# Test 3 — Frequency-dependent boundary effectiveness
# ======================================================================
def test_freq_boundary():
    print('\n=== Test 3: Frequency-weighted boundary — low vs high freq ===')
    f0 = 3e6; c0 = 1500; lam = c0 / f0
    dX = lam / 5; dT = dX / (5 * c0)
    nX = 61; nY = 61; nT = 101

    # Test at two frequencies: f0/3 (low) and f0 (center)
    results = {}
    for freq_label, freq in [('low_freq', f0/3), ('center_freq', f0)]:
        wX = 6e-3; wY = 6e-3; p0 = 1e3
        apa, xaxis, yaxis, t = _make_ic(nX, nY, nT, dX, dT, freq, c0, p0, wX, wY)

        for bdy_label, use_fw in [('standard', False), ('freq_weighted', True)]:
            params = SolverParams(
                dX=dX, dY=dX, dT=dT, c0=c0, rho0=1000, beta=0.0,
                alpha0=-1, f0=f0, propDist=2e-2, boundaryFactor=0.15,
                useSplitStep=True, useAdaptiveFiltering=False, useTVD=False,
                boundaryProfile='wendland',
                useFreqWeightedBoundary=use_fw,
                useSuperAbsorbing=False, dZmin=dX * 4)
            field, _, _, pI, _, zaxis, _ = angular_spectrum_solve(
                apa, params, verbose=False)

            key = f'{freq_label}_{bdy_label}'
            # Energy at final step
            I_final = np.sum(pI[:, :, -1])
            I_initial = np.sum(pI[:, :, 0])
            retention = I_final / (I_initial + 1e-30)
            results[key] = dict(pI=pI, zaxis=zaxis, retention=retention)
            print(f'  {key:35s}: energy retention = {retention:.4f}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for freq_label, freq, ax in [('low_freq', f0/3, axes[0]),
                                  ('center_freq', f0, axes[1])]:
        for bdy_label, ls in [('standard', '-'), ('freq_weighted', '--')]:
            key = f'{freq_label}_{bdy_label}'
            r = results[key]
            z = r['zaxis']
            I_vs_z = np.sum(r['pI'], axis=(0,1))
            ax.plot(z * 1e3, I_vs_z / I_vs_z[0], ls, label=bdy_label)
        ax.set(xlabel='z (mm)', ylabel='Normalized total intensity',
               title=f'{freq_label} ({freq/1e6:.1f} MHz)')
        ax.legend(); ax.grid(True)

    fig.suptitle('Frequency-weighted boundary: energy retention', fontsize=14)
    plt.tight_layout()
    _save_fig(fig, 'test3_freq_boundary.png')
    return results


# ======================================================================
# Test 4 — Super-absorbing boundary directional test
# ======================================================================
def test_super_absorbing():
    print('\n=== Test 4: Super-absorbing boundary ===')
    f0 = 3e6; c0 = 1500; lam = c0 / f0
    dX = lam / 5; dT = dX / (5 * c0)
    nX = 61; nY = 61; nT = 101
    wX = 7e-3; wY = 7e-3; p0 = 1e3

    apa, xaxis, yaxis, t = _make_ic(nX, nY, nT, dX, dT, f0, c0, p0, wX, wY)

    strengths = [0.0, 0.3, 0.6, 0.9]
    results = {}
    for s in strengths:
        params = SolverParams(
            dX=dX, dY=dX, dT=dT, c0=c0, rho0=1000, beta=0.0,
            alpha0=-1, f0=f0, propDist=2e-2, boundaryFactor=0.15,
            useSplitStep=True, useAdaptiveFiltering=False, useTVD=False,
            boundaryProfile='wendland',
            useFreqWeightedBoundary=False,
            useSuperAbsorbing=s > 0,
            superAbsorbingStrength=s,
            dZmin=dX * 4)
        field, _, _, pI, _, zaxis, pax = angular_spectrum_solve(
            apa, params, verbose=False)
        results[s] = dict(pI=pI, zaxis=zaxis, pax=pax, field=field)
        max_bdy = np.max(np.abs(field[0, :, :]))  # field at left edge
        print(f'  strength={s:.1f}: max field at boundary = {max_bdy:.1f} Pa')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for s in strengths:
        r = results[s]
        z = r['zaxis']
        I_vs_z = np.sum(r['pI'], axis=(0,1))
        axes[0].plot(z * 1e3, I_vs_z, label=f's={s:.1f}')
    axes[0].set(xlabel='z (mm)', ylabel='Total intensity',
                title='Super-absorbing: total intensity vs z')
    axes[0].legend(); axes[0].grid(True)

    # Final waveform comparison
    for s in strengths:
        axes[1].plot(t * 1e6, results[s]['pax'][:, -1], label=f's={s:.1f}')
    axes[1].set(xlabel='t (us)', ylabel='Pa',
                title='Super-absorbing: final axial waveform')
    axes[1].legend(); axes[1].grid(True)

    _save_fig(fig, 'test4_super_absorbing.png')
    return results


# ======================================================================
# Test 5 — Full comparison: all improvements combined
# ======================================================================
def test_combined():
    print('\n=== Test 5: Combined improvements vs baseline ===')
    f0 = 3e6; c0 = 1500; lam = c0 / f0
    dX = lam / 5; dT = dX / (5 * c0)
    nX = 71; nY = 71; nT = 121
    wX = 8e-3; wY = 8e-3; p0 = 0.03e6

    apa, xaxis, yaxis, t = _make_ic(nX, nY, nT, dX, dT, f0, c0, p0, wX, wY)

    configs = [
        ('baseline (quadratic)',
         dict(boundaryProfile='quadratic', useFreqWeightedBoundary=False,
              useSuperAbsorbing=False)),
        ('wendland only',
         dict(boundaryProfile='wendland', useFreqWeightedBoundary=False,
              useSuperAbsorbing=False)),
        ('wendland + freq-weighted',
         dict(boundaryProfile='wendland', useFreqWeightedBoundary=True,
              useSuperAbsorbing=False)),
        ('all combined',
         dict(boundaryProfile='wendland', useFreqWeightedBoundary=True,
              useSuperAbsorbing=True, superAbsorbingStrength=0.5)),
    ]

    results = {}
    for label, kw in configs:
        params = SolverParams(
            dX=dX, dY=dX, dT=dT, c0=c0, rho0=1000, beta=3.5,
            alpha0=0.5, attenPow=1, f0=f0, propDist=3e-2,
            boundaryFactor=0.15, useSplitStep=True,
            useAdaptiveFiltering=False, useTVD=False,
            dZmin=dX * 4, **kw)
        field, pnp, ppp, pI, pIloss, zaxis, pax = angular_spectrum_solve(
            apa, params, verbose=False)
        results[label] = dict(field=field, pI=pI, zaxis=zaxis, pax=pax)
        max_val = np.max(np.abs(field))
        print(f'  {label:30s}: max |p| = {max_val:.1f} Pa')

    # ------------------------------------------------------------------
    # Main figure: xz intensity maps + (config − baseline) difference maps
    # ------------------------------------------------------------------
    labels = [c[0] for c in configs]
    baseline_label = labels[0]

    # Boundary-onset coordinates so readers can see where ABL kicks in.
    boundary_factor = 0.15
    n_bdy_x = int(round(nX * boundary_factor))
    x_edge_lo = xaxis[n_bdy_x] * 1e3
    x_edge_hi = xaxis[nX - n_bdy_x - 1] * 1e3

    z0 = results[baseline_label]['zaxis']
    pI_xz = {lab: np.asarray(results[lab]['pI'][:, nY // 2, :]) for lab in labels}
    base_xz = pI_xz[baseline_label]
    peak_any = max(pI_xz[lab].max() for lab in labels) + 1e-30

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    z_extent = [0, z0[-1] * 1e3]
    x_extent = [xaxis[0] * 1e3, xaxis[-1] * 1e3]

    # Row 1: log-scale intensity xz maps (0 dB = overall peak)
    for col, lab in enumerate(labels):
        pI_db = 10.0 * np.log10(pI_xz[lab] / peak_any + 1e-12)
        im = axes[0, col].imshow(
            pI_db, aspect='auto', origin='lower',
            extent=[z_extent[0], z_extent[1], x_extent[0], x_extent[1]],
            vmin=-60, vmax=0, cmap='viridis')
        axes[0, col].axhline(x_edge_lo, color='w', ls=':', lw=1)
        axes[0, col].axhline(x_edge_hi, color='w', ls=':', lw=1)
        axes[0, col].set(xlabel='z (mm)', ylabel='x (mm)', title=lab)
        plt.colorbar(im, ax=axes[0, col], label='dB vs peak')

    # Row 2: (config − baseline) difference.  Symmetric diverging cmap.
    diff_arrays = [pI_xz[lab] - base_xz for lab in labels]
    diff_peak = max(np.max(np.abs(d)) for d in diff_arrays) + 1e-30
    for col, (lab, d) in enumerate(zip(labels, diff_arrays)):
        im = axes[1, col].imshow(
            d, aspect='auto', origin='lower',
            extent=[z_extent[0], z_extent[1], x_extent[0], x_extent[1]],
            vmin=-diff_peak, vmax=diff_peak, cmap='RdBu_r')
        axes[1, col].axhline(x_edge_lo, color='k', ls=':', lw=1)
        axes[1, col].axhline(x_edge_hi, color='k', ls=':', lw=1)
        title = ('reference (zero by defn.)' if col == 0
                 else f'{lab} − baseline')
        axes[1, col].set(xlabel='z (mm)', ylabel='x (mm)', title=title)
        plt.colorbar(im, ax=axes[1, col])

    fig.suptitle('Test 10: boundary configurations — xz intensity (top) and '
                 '(config − baseline) difference (bottom); dotted lines mark '
                 'the ABL onset', fontsize=12)
    plt.tight_layout()
    _save_fig(fig, 'test5_combined.png')

    # ------------------------------------------------------------------
    # Companion summary: final axial waveform + spectrum only
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5))

    for label, r in results.items():
        ax2[0].plot(t * 1e6, r['pax'][:, -1], label=label)
    ax2[0].set(xlabel='t (µs)', ylabel='Pa',
               title='Final axial waveform')
    ax2[0].legend(fontsize=8); ax2[0].grid(True)

    for label, r in results.items():
        spec = np.abs(np.fft.rfft(r['pax'][:, -1]))
        freqs = np.fft.rfftfreq(len(t), dT) / 1e6
        ax2[1].plot(freqs, 20 * np.log10(spec / (np.max(spec) + 1e-30) + 1e-30),
                    label=label)
    ax2[1].set(xlabel='f (MHz)', ylabel='dB',
               title='Final waveform spectrum')
    ax2[1].set_xlim([0, 4 * f0 / 1e6])
    ax2[1].legend(fontsize=8); ax2[1].grid(True)

    plt.tight_layout()
    _save_fig(fig2, 'test5_combined_profiles.png')

    return results


# ======================================================================
if __name__ == '__main__':
    print('Boundary Condition Validation Suite')
    print('=' * 60)

    test_profiles()
    test_reflection()
    test_freq_boundary()
    test_super_absorbing()
    test_combined()

    print('\n' + '=' * 60)
    print('All boundary tests complete. Results in validation_results/boundaries/')
