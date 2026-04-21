"""
Test 11f — Transcranial k-space filter showcase.

Reuses the 1 MHz, 50 mm-focus sparse-array transcranial setup from
validate_transcranial.py, but runs the ASM through the same phase-screen
stack with the adaptive k-space filter ON and OFF, holding all other
solver options fixed.  Also runs a homogeneous-water reference at the
same grid for a loss-vs-skull comparison.

Skull-bone microstructure injects high transverse wavenumbers on every
screen, and the nonlinear operator (beta = 3.5) mixes modes across
steps.  The filter's role is to prevent this accumulated high-k content
from aliasing back into the physical band.

PPW (points per wavelength) is a CLI argument.  At PPW = 5 (default
λ/5) the rolloff band sits above k0 and the filter does nothing; at
PPW = 2.5 it clips the highest-angle propagating modes and should
matter.  Each PPW gets its own cache directory.

Usage:
  python validate_kfilter_transcranial.py                 # PPW = 5
  python validate_kfilter_transcranial.py --ppw 2.5       # stress test
  python validate_kfilter_transcranial.py --ppw 3.3 --force-rerun
"""

import os, sys, time, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from angular_spectrum_solver import SolverParams, angular_spectrum_solve
from validate_transcranial import (
    load_skull_slab, skull_to_phase_screens,
    load_sparse_connector, build_initial_condition,
    NRRD_PATH, CONNECTOR_PATH,
)

OUTDIR = os.path.join(os.path.dirname(__file__), 'validation_results', 'kfilter')
os.makedirs(OUTDIR, exist_ok=True)


def _save_fig(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def _build_setup(ppw=5):
    """Build transducer, skull slab, and screens at the requested PPW.

    ppw = wavelengths per transverse grid cell (dX = lambda / ppw).
    Lower ppw -> coarser grid -> larger fraction of k-space inside
    the filter's rolloff band.
    """
    f0 = 1e6
    c0 = 1540.0
    rho0 = 1000.0
    lam = c0 / f0
    p0 = 1.5e6
    focal_depth = 50e-3
    domain_depth = 65e-3
    domain_lat = 66e-3

    dX = lam / ppw
    dT = dX / (5 * c0)
    nX = int(round(domain_lat / dX))
    nY = nX
    if nX % 2 == 0:
        nX += 1
        nY += 1

    ncycles = 2
    duration_min = ncycles * 6 / f0
    delay_spread_est = 15e-6
    duration = 2.0 * (duration_min + delay_spread_est)
    nT = int(round(duration / dT))
    if nT % 2 == 0:
        nT += 1
    nT = min(nT, 501)

    xaxis = np.arange(nX) * dX - (nX - 1) * dX / 2
    yaxis = np.arange(nY) * dX - (nY - 1) * dX / 2
    t = np.arange(nT) * dT
    t -= np.mean(t)
    dZ = dX

    # Diagnostic: filter rolloff band relative to k0
    k0 = 2 * np.pi * f0 / c0
    k_nyq = np.pi / dX
    print(f'Grid: {nX}x{nY}x{nT}  dX={dX*1e3:.3f} mm  dT={dT*1e9:.2f} ns')
    print(f'  k_Nyq / k0 = {k_nyq/k0:.3f}  '
          f'(propagating band: |k_perp| <= k0 is '
          f'{100*k0/k_nyq:.0f}% of Nyquist)')
    print(f'  default filter rolloff: 76-95% of Nyquist '
          f'= {0.76*k_nyq/k0:.3f}-{0.95*k_nyq/k0:.3f} k0')

    positions, element_ids = load_sparse_connector(CONNECTOR_PATH)
    apa = build_initial_condition(positions, element_ids, xaxis, yaxis, t,
                                  f0, c0, p0, focal_depth, ncycles=ncycles)

    placement = {
        'xdc_center_lps': (140.6, 118.5, 85.0),
        'normal': (-0.954, -0.298),
        'tangent': (-0.298, 0.954),
        'z_mm': 85.0,
    }
    slab_size_m = (domain_lat, domain_lat, domain_depth)
    c_map, rho_map, dz_screen = load_skull_slab(
        NRRD_PATH, placement, slab_size_m, dX)

    from scipy.ndimage import zoom
    if c_map.shape[0] != nX or c_map.shape[1] != nY:
        zoom_factors = (nX / c_map.shape[0], nY / c_map.shape[1], 1.0)
        c_map = zoom(c_map, zoom_factors, order=1)
        rho_map = zoom(rho_map, zoom_factors, order=1)

    screens = skull_to_phase_screens(c_map, dz_screen, dZ, c0, f0,
                                     rho_map=rho_map, rho0=rho0,
                                     alpha_bone=8.0, alpha_tissue=0.5)

    return dict(
        apa=apa, screens=screens,
        xaxis=xaxis, yaxis=yaxis, dT=dT, dX=dX, dZ=dZ,
        nX=nX, nY=nY, nT=nT,
        f0=f0, c0=c0, rho0=rho0, p0=p0,
        focal_depth=focal_depth, domain_depth=domain_depth,
        ppw=ppw, k_nyq_over_k0=k_nyq/k0,
    )


def _run_config(setup, use_filter, use_screens, label):
    """Run one configuration (filter flag x screens on/off)."""
    print(f'\n=== Running: {label} ===')
    params = SolverParams(
        dX=setup['dX'], dY=setup['dX'], dT=setup['dT'],
        c0=setup['c0'], rho0=setup['rho0'], beta=3.5,
        alpha0=0.5, attenPow=1, f0=setup['f0'],
        propDist=setup['domain_depth'],
        boundaryFactor=0.12, useSplitStep=True,
        useAdaptiveFiltering=use_filter,
        useTVD=False, fluxScheme='kt',
        boundaryProfile='wendland',
        useFreqWeightedBoundary=True,
        useSuperAbsorbing=True, superAbsorbingStrength=0.5,
        stabilityThreshold=0.2, stabilityRecoveryFactor=0.15,
        dZmin=setup['dZ'],
        phaseScreens=setup['screens'] if use_screens else None)

    t0 = time.time()
    field, pnp, ppp, pI, pIloss, zaxis, pax = angular_spectrum_solve(
        setup['apa'], params, verbose=True)
    dt_run = time.time() - t0

    nan_count = int(np.sum(~np.isfinite(pI)))
    peak_pI = float(np.nanmax(np.where(np.isfinite(pI), pI, -np.inf)))
    print(f'\n  [{label}] runtime={dt_run:.1f}s  non-finite={nan_count}  '
          f'peak_pI={peak_pI:.3e}')
    return dict(pI=pI, pnp=pnp, ppp=ppp, zaxis=zaxis, pax=pax,
                runtime_s=dt_run, nan_count=nan_count, peak_pI=peak_pI)


def _norm_dB(arr, floor_dB=-40):
    arr_pos = np.maximum(arr, 0)
    pk = np.max(arr_pos) + 1e-30
    return np.maximum(10*np.log10(arr_pos/pk + 1e-30), floor_dB)


def main(ppw=5.0, force_rerun=False):
    tag = f'ppw{ppw:.2f}'.rstrip('0').rstrip('.')
    print(f'Test 11f: Transcranial k-space filter (ON vs OFF) at PPW = {ppw}')
    print('=' * 60)

    cache_path = os.path.join(OUTDIR, f'test11f_cache_{tag}.npz')

    if os.path.exists(cache_path) and not force_rerun:
        print(f'\nLoading cache: {cache_path}')
        d = np.load(cache_path)
        pI_homo = d['pI_homo']; pI_on = d['pI_on']; pI_off = d['pI_off']
        xaxis = d['xaxis']; yaxis = d['yaxis']; zaxis = d['zaxis']
        runtime_homo = float(d['runtime_homo'])
        runtime_on = float(d['runtime_on'])
        runtime_off = float(d['runtime_off'])
        nan_off = int(d['nan_off'])
        k_nyq_over_k0 = float(d['k_nyq_over_k0'])
    else:
        setup = _build_setup(ppw=ppw)
        xaxis = setup['xaxis']; yaxis = setup['yaxis']
        k_nyq_over_k0 = setup['k_nyq_over_k0']

        r_homo = _run_config(setup, use_filter=True,  use_screens=False,
                             label='homo (filter ON, no screens)')
        r_on   = _run_config(setup, use_filter=True,  use_screens=True,
                             label=f'skull filter ON')
        r_off  = _run_config(setup, use_filter=False, use_screens=True,
                             label=f'skull filter OFF')

        # Sanity: all three share zaxis
        zaxis = r_homo['zaxis']
        pI_homo = r_homo['pI']
        pI_on   = r_on['pI']
        pI_off  = r_off['pI']
        runtime_homo = r_homo['runtime_s']
        runtime_on   = r_on['runtime_s']
        runtime_off  = r_off['runtime_s']
        nan_off = r_off['nan_count']

        np.savez(cache_path,
                 pI_homo=pI_homo, pI_on=pI_on, pI_off=pI_off,
                 xaxis=xaxis, yaxis=yaxis, zaxis=zaxis,
                 runtime_homo=runtime_homo,
                 runtime_on=runtime_on,
                 runtime_off=runtime_off,
                 nan_off=nan_off,
                 k_nyq_over_k0=k_nyq_over_k0,
                 ppw=ppw)
        print(f'\n  saved cache: {cache_path}')

    nX, nY, nZ = pI_on.shape

    # Replace non-finite with zeros for analysis (but keep nan_off flag)
    pI_on_s   = np.where(np.isfinite(pI_on),   pI_on,   0.0)
    pI_off_s  = np.where(np.isfinite(pI_off),  pI_off,  0.0)
    pI_homo_s = np.where(np.isfinite(pI_homo), pI_homo, 0.0)

    # Focal-region metrics (z > 30 mm)
    far_mask = zaxis > 30e-3
    shift = int(np.argmax(far_mask))
    focal_idx_homo = shift + int(np.argmax(pI_homo_s[nX//2, nY//2, far_mask]))
    focal_idx_on   = shift + int(np.argmax(pI_on_s  [nX//2, nY//2, far_mask]))
    focal_idx_off  = shift + int(np.argmax(pI_off_s [nX//2, nY//2, far_mask]))

    peak_homo = float(np.max(pI_homo_s))
    peak_on   = float(np.max(pI_on_s[:, :, far_mask]))
    peak_off  = float(np.max(pI_off_s[:, :, far_mask]))
    loss_on_dB  = 10*np.log10(peak_on  / (peak_homo+1e-30) + 1e-30)
    loss_off_dB = 10*np.log10(peak_off / (peak_homo+1e-30) + 1e-30)

    # Out-of-lobe floor at each run's focal plane (normalized to each peak)
    XX, YY = np.meshgrid(xaxis, yaxis, indexing='ij')
    out_mask = (np.abs(XX) > 8e-3) | (np.abs(YY) > 8e-3)
    def _side(arr2d):
        pk = np.max(arr2d) + 1e-30
        return 10*np.log10(np.max(arr2d[out_mask]) / pk + 1e-30)
    side_on  = _side(pI_on_s [:, :, focal_idx_on])
    side_off = _side(pI_off_s[:, :, focal_idx_off])

    # Normalized dB focal plane
    on_focal_ndB  = _norm_dB(pI_on_s [:, :, focal_idx_on])
    off_focal_ndB = _norm_dB(pI_off_s[:, :, focal_idx_off])
    dB_rms_diff = float(np.sqrt(np.mean((on_focal_ndB - off_focal_ndB)**2)))

    print('\n' + '-'*74)
    print(f'  PPW = {ppw}    k_Nyq / k0 = {k_nyq_over_k0:.3f}')
    print('-'*74)
    print(f'{"metric":<34s} {"filter ON":>14s} {"filter OFF":>14s} {"homo":>10s}')
    print('-'*74)
    print(f'{"focal z (mm)":<34s} {zaxis[focal_idx_on]*1e3:>14.2f} '
          f'{zaxis[focal_idx_off]*1e3:>14.2f} {zaxis[focal_idx_homo]*1e3:>10.2f}')
    print(f'{"focal loss vs homo (dB)":<34s} {loss_on_dB:>14.2f} '
          f'{loss_off_dB:>14.2f} {0.00:>10.2f}')
    print(f'{"out-of-lobe floor (dB)":<34s} {side_on:>14.2f} '
          f'{side_off:>14.2f} {"—":>10s}')
    print(f'{"peak pI":<34s} {peak_on:>14.3e} {peak_off:>14.3e} {peak_homo:>10.3e}')
    print(f'{"non-finite voxels":<34s} {0:>14d} {nan_off:>14d} {0:>10d}')
    print(f'{"runtime (s)":<34s} {runtime_on:>14.1f} {runtime_off:>14.1f} {runtime_homo:>10.1f}')
    print('-'*74)
    print(f'  dB focal-plane RMS diff (ON vs OFF):  {dB_rms_diff:.3f} dB')

    # --- Figure 1: side-by-side dB maps (x-z and focal x-y) ---
    ext_xz = [xaxis[0]*1e3, xaxis[-1]*1e3, zaxis[-1]*1e3, 0]
    ext_xy = [xaxis[0]*1e3, xaxis[-1]*1e3, yaxis[0]*1e3, yaxis[-1]*1e3]

    on_xz_dB  = _norm_dB(pI_on_s [:, nY//2, :].T)
    off_xz_dB = _norm_dB(pI_off_s[:, nY//2, :].T)
    diff_xz_dB = off_xz_dB - on_xz_dB

    on_xy_dB  = on_focal_ndB.T
    off_xy_dB = off_focal_ndB.T
    diff_xy_dB = off_xy_dB - on_xy_dB

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    im = axes[0, 0].imshow(on_xz_dB, aspect='auto', extent=ext_xz,
                           origin='upper', vmin=-40, vmax=0, cmap='inferno')
    axes[0, 0].set(xlabel='x (mm)', ylabel='Depth (mm)',
                   title='x-z  filter ON (dB)')
    plt.colorbar(im, ax=axes[0, 0])

    axes[0, 1].imshow(off_xz_dB, aspect='auto', extent=ext_xz,
                      origin='upper', vmin=-40, vmax=0, cmap='inferno')
    axes[0, 1].set(xlabel='x (mm)', ylabel='Depth (mm)',
                   title='x-z  filter OFF (dB)')

    vlim = max(abs(np.nanmin(diff_xz_dB)), abs(np.nanmax(diff_xz_dB)), 1.0)
    im_d = axes[0, 2].imshow(diff_xz_dB, aspect='auto', extent=ext_xz,
                             origin='upper', vmin=-vlim, vmax=vlim,
                             cmap='RdBu_r')
    axes[0, 2].set(xlabel='x (mm)', ylabel='Depth (mm)',
                   title='x-z  OFF - ON (dB)')
    plt.colorbar(im_d, ax=axes[0, 2])

    axes[1, 0].imshow(on_xy_dB, extent=ext_xy, aspect='equal',
                      vmin=-40, vmax=0, cmap='inferno')
    axes[1, 0].set(xlabel='x (mm)', ylabel='y (mm)',
                   title=f'x-y @ z={zaxis[focal_idx_on]*1e3:.1f} mm  ON')

    axes[1, 1].imshow(off_xy_dB, extent=ext_xy, aspect='equal',
                      vmin=-40, vmax=0, cmap='inferno')
    axes[1, 1].set(xlabel='x (mm)', ylabel='y (mm)',
                   title=f'x-y @ z={zaxis[focal_idx_off]*1e3:.1f} mm  OFF')

    vlim2 = max(abs(np.nanmin(diff_xy_dB)), abs(np.nanmax(diff_xy_dB)), 1.0)
    im_d2 = axes[1, 2].imshow(diff_xy_dB, extent=ext_xy, aspect='equal',
                              vmin=-vlim2, vmax=vlim2, cmap='RdBu_r')
    axes[1, 2].set(xlabel='x (mm)', ylabel='y (mm)',
                   title='x-y  OFF - ON (dB)')
    plt.colorbar(im_d2, ax=axes[1, 2])

    fig.suptitle(f'Transcranial ASM: k-space filter ON vs OFF  '
                 f'(PPW = {ppw}, k_Nyq/k0 = {k_nyq_over_k0:.2f})',
                 fontsize=13)
    _save_fig(fig, f'test11f_transcranial_kfilter_{tag}.png')

    # --- Figure 2: on-axis + lateral + elevation profiles ---
    fig2, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ref_peak = np.max(pI_on_s[nX//2, nY//2, :]) + 1e-30
    onaxis_on_abs  = 10*np.log10(pI_on_s [nX//2, nY//2, :] / ref_peak + 1e-30)
    onaxis_off_abs = 10*np.log10(pI_off_s[nX//2, nY//2, :] / ref_peak + 1e-30)
    onaxis_h_abs   = 10*np.log10(pI_homo_s[nX//2, nY//2, :] / (np.max(pI_homo_s[nX//2, nY//2, :])+1e-30) + 1e-30)
    axes[0].plot(zaxis*1e3, onaxis_h_abs,   'C2:',  lw=1.2, label='homo (filter ON)')
    axes[0].plot(zaxis*1e3, onaxis_on_abs,  'C0-',  lw=1.4, label='skull filter ON')
    axes[0].plot(zaxis*1e3, onaxis_off_abs, 'C3--', lw=1.4, label='skull filter OFF')
    axes[0].set(xlabel='Depth (mm)', ylabel='On-axis intensity (dB, each re own peak)',
                title='On-axis profile')
    axes[0].legend(fontsize=9); axes[0].grid(True)
    axes[0].set_ylim(-40, 5)

    lat_on  = pI_on_s [:, nY//2, focal_idx_on]
    lat_off = pI_off_s[:, nY//2, focal_idx_off]
    lat_on_dB  = 10*np.log10(lat_on  / (np.max(lat_on) +1e-30) + 1e-30)
    lat_off_dB = 10*np.log10(lat_off / (np.max(lat_off)+1e-30) + 1e-30)
    axes[1].plot(xaxis*1e3, lat_on_dB,  'C0-', lw=1.4, label='filter ON')
    axes[1].plot(xaxis*1e3, lat_off_dB, 'C3--', lw=1.4, label='filter OFF')
    axes[1].set(xlabel='x (mm)', ylabel='Intensity (dB re own peak)',
                title='Lateral profile at focus')
    axes[1].legend(); axes[1].grid(True)
    axes[1].set_ylim(-40, 2)

    elev_on  = pI_on_s [nX//2, :, focal_idx_on]
    elev_off = pI_off_s[nX//2, :, focal_idx_off]
    elev_on_dB  = 10*np.log10(elev_on  / (np.max(elev_on) +1e-30) + 1e-30)
    elev_off_dB = 10*np.log10(elev_off / (np.max(elev_off)+1e-30) + 1e-30)
    axes[2].plot(yaxis*1e3, elev_on_dB,  'C0-', lw=1.4, label='filter ON')
    axes[2].plot(yaxis*1e3, elev_off_dB, 'C3--', lw=1.4, label='filter OFF')
    axes[2].set(xlabel='y (mm)', ylabel='Intensity (dB re own peak)',
                title='Elevation profile at focus')
    axes[2].legend(); axes[2].grid(True)
    axes[2].set_ylim(-40, 2)

    fig2.suptitle(f'Transcranial ASM profiles (PPW = {ppw})', fontsize=12)
    plt.tight_layout()
    _save_fig(fig2, f'test11f_transcranial_onaxis_{tag}.png')

    np.savez(os.path.join(OUTDIR, f'test11f_transcranial_summary_{tag}.npz'),
             ppw=ppw, k_nyq_over_k0=k_nyq_over_k0,
             xaxis=xaxis, yaxis=yaxis, zaxis=zaxis,
             focal_z_on=zaxis[focal_idx_on],
             focal_z_off=zaxis[focal_idx_off],
             focal_z_homo=zaxis[focal_idx_homo],
             focal_loss_on_dB=loss_on_dB,
             focal_loss_off_dB=loss_off_dB,
             side_on_dB=side_on, side_off_dB=side_off,
             dB_rms_diff=dB_rms_diff,
             nan_off=nan_off,
             peak_on=peak_on, peak_off=peak_off, peak_homo=peak_homo,
             runtime_on=runtime_on, runtime_off=runtime_off, runtime_homo=runtime_homo)
    print(f'\n  saved summary')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ppw', type=float, default=5.0,
                    help='Points per wavelength (transverse). Default 5.')
    ap.add_argument('--force-rerun', action='store_true',
                    help='Ignore existing cache and rerun from scratch.')
    args = ap.parse_args()
    main(ppw=args.ppw, force_rerun=args.force_rerun)
