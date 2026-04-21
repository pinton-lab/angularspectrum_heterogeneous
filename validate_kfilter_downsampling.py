"""
Test 11e — PPW down-sampling stress test for the k-space filter.

Fixes the physical problem (focused piston, a = 8 lambda, F = 20 mm,
linear regime) and sweeps transverse sampling from well-resolved
(PPW ~ 8) down to coarse (PPW ~ 3), where the discrete transverse
spectrum of the aperture's diffraction pattern begins to populate
wavenumbers that alias through the Nyquist.  At each PPW the solver is
run twice — with and without the adaptive k-space filter — and
compared against the paraxial focal limit

    |p(0, F)| = p0 * pi * a^2 / (lambda * F)

and the focal-plane Airy-like lateral pattern (2 J1(u)/u)^2.

The expected signature: the unfiltered curve degrades sharply (often
non-monotonically, with aliasing replicas) as PPW drops, while the
filtered curve degrades gracefully and monotonically.

Results saved to validation_results/kfilter/.
"""

import os, sys, time
import numpy as np
from scipy.special import j1 as besselj1
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from angular_spectrum_solver import SolverParams, angular_spectrum_solve

OUTDIR = os.path.join(os.path.dirname(__file__), 'validation_results', 'kfilter')
os.makedirs(OUTDIR, exist_ok=True)


def _save_fig(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def _run_one(ppw, use_filter, f0, c0, rho0, a, F_foc, p0,
             nX_target=193, ncycles=4.0):
    """Run a single focused-piston propagation at given PPW / filter setting.

    Returns focal metrics and lateral slice for plotting.
    """
    lam = c0 / f0
    dX = lam / ppw
    dT = dX / (5 * c0)

    # Keep transverse physical extent roughly constant across PPW:
    # width = nX_target * (lam/5) in the reference configuration.
    ref_width = nX_target * (lam / 5.0)
    nX = int(round(ref_width / dX))
    if nX % 2 == 0:
        nX += 1            # keep an on-axis grid point
    nY = nX
    nT = 1201

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

    tau = (F_foc - np.sqrt(F_foc**2 + RR2)) / c0
    delay_samples = np.round(tau / dT).astype(np.int32)

    field0 = np.zeros((nX, nY, nT), dtype=np.float32)
    xs, ys = np.where(aperture > 0)
    for (ix, iy) in zip(xs, ys):
        field0[ix, iy, :] = np.roll(icvec, int(delay_samples[ix, iy]))

    propDist = 1.1 * F_foc
    params = SolverParams(
        dX=dX, dY=dX, dT=dT, c0=c0, rho0=rho0,
        beta=0.0, alpha0=-1, f0=f0, propDist=propDist,
        useSplitStep=False, useTVD=False,
        useAdaptiveFiltering=use_filter,
        boundaryFactor=0.15, useBoundaryLayer=True,
        stabilityThreshold=0.2, stabilityRecoveryFactor=0.15,
        dZmin=2 * dX,
    )
    t0 = time.time()
    field_out, pnp, ppp, pI, pIloss, zaxis, pax = \
        angular_spectrum_solve(field0, params, verbose=False)
    dt_run = time.time() - t0
    zaxis = np.asarray(zaxis)
    pax_np = np.asarray(pax)

    # On-axis peak amplitude vs z
    numeric_onaxis = np.max(np.abs(pax_np), axis=0)
    idx_F = int(np.argmin(np.abs(zaxis - F_foc)))

    # Focal-plane lateral slice along x from pI (time-integrated p^2)
    I_lat = np.asarray(pI[:, nY // 2, idx_F]).astype(np.float64)
    I_lat_norm = I_lat / (np.max(I_lat) + 1e-30)

    # Analytical references
    focal_amp = p0 * np.pi * a**2 / (lam * F_foc)
    focal_err = abs(numeric_onaxis[idx_F] - focal_amp) / focal_amp

    k0 = 2 * np.pi * f0 / c0
    u = k0 * a * np.abs(xaxis) / F_foc
    u_safe = np.where(u < 1e-10, 1e-10, u)
    model_focal = (2 * besselj1(u_safe) / u_safe)**2
    model_focal[u < 1e-10] = 1.0
    model_focal /= np.max(model_focal)
    lateral_rms = float(np.sqrt(np.mean((I_lat_norm - model_focal)**2)))

    # Sidelobe floor: max of normalized intensity outside the mainlobe
    # (|x| > 1 mm captures the first null for this geometry).
    sidelobe_mask = np.abs(xaxis) > 1e-3
    sidelobe_max = float(np.max(I_lat_norm[sidelobe_mask])) if sidelobe_mask.any() else 0.0

    return dict(
        ppw=ppw, use_filter=use_filter, dX=dX, nX=nX,
        zaxis=zaxis, numeric_onaxis=numeric_onaxis,
        xaxis=xaxis, I_lat_norm=I_lat_norm, model_focal=model_focal,
        focal_amp=focal_amp, p_at_F=float(numeric_onaxis[idx_F]),
        focal_err=focal_err, lateral_rms=lateral_rms,
        sidelobe_max=sidelobe_max, runtime_s=dt_run,
    )


def run_ppw_sweep(ppw_list=(8, 6, 5, 4, 3.5, 3), p0=0.3e6, tag=''):
    """Sweep points-per-wavelength, filter on vs off."""
    print(f'\n=== Test 11e: PPW down-sampling — k-filter on vs off '
          f'(PPW = {list(ppw_list)}) ===')

    f0 = 4e6
    c0 = 1500.0
    rho0 = 1000.0
    lam = c0 / f0
    a = 8 * lam
    F_foc = 20e-3

    rows = []
    lateral_cache = {}   # (ppw, use_filter) -> (xaxis, I_lat_norm, model)
    for ppw in ppw_list:
        for use_filter in (False, True):
            label = 'ON ' if use_filter else 'OFF'
            print(f'\n--- PPW = {ppw}, filter {label} ---')
            r = _run_one(ppw, use_filter, f0, c0, rho0, a, F_foc, p0)
            rows.append(r)
            lateral_cache[(ppw, use_filter)] = (r['xaxis'], r['I_lat_norm'],
                                                r['model_focal'])
            print(f'    dX              = {r["dX"]*1e6:.1f} µm  (nX = {r["nX"]})')
            print(f'    |p(F)| numeric  = {r["p_at_F"]:.3e} Pa')
            print(f'    |p(F)| frac err = {r["focal_err"]:.4f}')
            print(f'    lateral RMS     = {r["lateral_rms"]:.4f}')
            print(f'    sidelobe max    = {r["sidelobe_max"]:.3f}')
            print(f'    runtime         = {r["runtime_s"]:.1f} s')

    # --- Figure 1: focal error + lateral RMS vs PPW ---
    ppws = np.array(ppw_list, dtype=float)
    err_off = np.array([r['focal_err'] for r in rows if not r['use_filter']])
    err_on  = np.array([r['focal_err'] for r in rows if r['use_filter']])
    rms_off = np.array([r['lateral_rms'] for r in rows if not r['use_filter']])
    rms_on  = np.array([r['lateral_rms'] for r in rows if r['use_filter']])
    side_off = np.array([r['sidelobe_max'] for r in rows if not r['use_filter']])
    side_on  = np.array([r['sidelobe_max'] for r in rows if r['use_filter']])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(ppws, err_off, 'C3-o', lw=1.4, label='filter OFF')
    axes[0].plot(ppws, err_on,  'C0-s', lw=1.4, label='filter ON')
    axes[0].invert_xaxis()
    axes[0].set(xlabel='PPW (c0 / f0 / dX)',
                ylabel='|p(F)| fractional error',
                title='Focal-amplitude error vs PPW')
    axes[0].set_yscale('log')
    axes[0].legend(); axes[0].grid(True, which='both', alpha=0.3)

    axes[1].plot(ppws, rms_off, 'C3-o', lw=1.4, label='filter OFF')
    axes[1].plot(ppws, rms_on,  'C0-s', lw=1.4, label='filter ON')
    axes[1].invert_xaxis()
    axes[1].set(xlabel='PPW', ylabel='Focal-plane RMS vs Airy',
                title='Lateral profile RMS vs PPW')
    axes[1].set_yscale('log')
    axes[1].legend(); axes[1].grid(True, which='both', alpha=0.3)

    axes[2].plot(ppws, side_off, 'C3-o', lw=1.4, label='filter OFF')
    axes[2].plot(ppws, side_on,  'C0-s', lw=1.4, label='filter ON')
    axes[2].invert_xaxis()
    axes[2].set(xlabel='PPW', ylabel='max normalized intensity, |x|>1mm',
                title='Out-of-mainlobe floor (aliasing indicator)')
    axes[2].set_yscale('log')
    axes[2].legend(); axes[2].grid(True, which='both', alpha=0.3)

    fig.suptitle('PPW down-sampling: k-space filter ON vs OFF '
                 f'(a = 8 lambda, F = 20 mm, f0 = {f0/1e6:.1f} MHz)',
                 fontsize=12)
    plt.tight_layout()
    _save_fig(fig, f'test11e_ppw_sweep{tag}.png')

    # --- Figure 2: lateral profiles at the coarsest PPW, filter on vs off ---
    ppw_worst = min(ppw_list)
    x_off, I_off, model = lateral_cache[(ppw_worst, False)]
    x_on,  I_on,  _     = lateral_cache[(ppw_worst, True)]
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.semilogy(x_off*1e3, I_off, 'C3-', lw=1.2, label='filter OFF')
    ax2.semilogy(x_on*1e3,  I_on,  'C0-', lw=1.2, label='filter ON')
    ax2.semilogy(x_off*1e3, np.maximum(model, 1e-6), 'k--', lw=1.0,
                 label='$(2 J_1/u)^2$')
    ax2.set_ylim(1e-5, 2.0)
    ax2.set_xlim(-5, 5)
    ax2.set(xlabel='x (mm)', ylabel='Normalized intensity',
            title=f'Focal-plane lateral at PPW = {ppw_worst} '
                  '(aliasing replicas visible with filter OFF)')
    ax2.legend(); ax2.grid(True, which='both', alpha=0.3)
    _save_fig(fig2, f'test11e_ppw_worst_lateral{tag}.png')

    # Save arrays
    np.savez(os.path.join(OUTDIR, f'test11e_ppw_sweep{tag}.npz'),
             ppws=ppws, focal_err_off=err_off, focal_err_on=err_on,
             lateral_rms_off=rms_off, lateral_rms_on=rms_on,
             sidelobe_off=side_off, sidelobe_on=side_on,
             a=a, F=F_foc, f0=f0, c0=c0)

    # Summary table
    print('\n  ' + '-'*74)
    print(f'  {"PPW":>5s} {"filter":>7s} {"|p(F)| err":>12s} '
          f'{"lat RMS":>10s} {"sidelobe":>10s} {"runtime":>10s}')
    print('  ' + '-'*74)
    for r in rows:
        print(f'  {r["ppw"]:>5.1f} {"ON" if r["use_filter"] else "OFF":>7s} '
              f'{r["focal_err"]:>12.4f} {r["lateral_rms"]:>10.4f} '
              f'{r["sidelobe_max"]:>10.4f} {r["runtime_s"]:>9.1f}s')
    print('  ' + '-'*74)

    return rows


if __name__ == '__main__':
    run_ppw_sweep()
