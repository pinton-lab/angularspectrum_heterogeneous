"""
Validate plane-by-plane bowl source (ASM) against FDTD for a TIPS transducer.

Water-only comparison: no skull, no nonlinearity — pure diffraction so any
discrepancy is due to the source modelling, not medium effects.

Produces two publication-quality figures:
  Figure A  — TIPS setup overview (bowl geometry, source plane, aperture)
  Figure B  — Three-way comparison: ASM flat, ASM plane-by-plane, FDTD

Compares:
  1. ASM with flat-projection bowl source  (make_bowl_source)
  2. ASM with plane-by-plane bowl source   (make_bowl_source_planes)
  3. FDTD with volumetric bowl source      (fullwave 3-D solver)
"""

import os
import sys
import time as _time
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from angular_spectrum_solver import (
    make_bowl_source,
    make_bowl_source_planes,
    SolverParams,
    angular_spectrum_solve,
)

OUTDIR = os.path.join(os.path.dirname(__file__), 'validation_results', 'bowl')
os.makedirs(OUTDIR, exist_ok=True)

# TIPS transducer parameters
F0 = 1e6
C0 = 1540.0
RHO0 = 1000.0
ROC = 80e-3          # radius of curvature
OUTER_R = 46e-3      # outer aperture radius
INNER_R = 20.5e-3    # inner radius (annular hole)
N_ELEMENTS = 8
# FDTD places the focus at 50mm from the inner edge of the bowl (x=0
# in the FDTD domain).  The inner edge sits at bowl depth z_inner =
# ROC - sqrt(ROC^2 - INNER_R^2) below the apex.  To match the FDTD
# convention, the ASM focal depth is measured from the apex:
_Z_INNER = ROC - np.sqrt(ROC**2 - INNER_R**2)  # ~2.67 mm
FOCAL_DEPTH = 50e-3 + _Z_INNER  # ~52.7 mm from apex
P0_FDTD = 4e5        # 400 kPa — FDTD source amplitude per layer
N_LAYERS = 3         # FDTD uses 3 voxel layers along surface normal
# Source-convention adjustment: FDTD uses an additive source condition
# (forcing added to the equation of motion over N_LAYERS voxel layers along
# the surface normal), while the ASM uses an imposed source condition
# (pressure prescribed on the source plane). The factor 1.306 aligns the
# two conventions so the focal pressures are directly comparable. The value
# reflects the coherent sum of the three-layer additive source at the FDTD
# grid spacing (k*dx ~ 0.79 rad, CFL = 0.2); adjust if N_LAYERS or the FDTD
# grid resolution change.
P0 = P0_FDTD * 1.306
NCYCLES = 3
DUR = 2              # super-Gaussian exponent


def _save_fig(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# ======================================================================
# FDTD
# ======================================================================
def run_fdtd():
    """Run 3-D FDTD simulation of TIPS transducer in water."""
    import fullwave
    from fullwave.transducers.tips import create_tips_transducer
    from fullwave.medium_builder.presets import BackgroundDomain
    from fullwave.medium_builder import MediumBuilder

    print('\n--- FDTD: TIPS in water ---')

    domain_depth = 80e-3
    domain_lat = 112e-3
    duration = domain_depth / C0 * 2

    grid = fullwave.Grid(
        (domain_depth, domain_lat, domain_lat),
        F0, duration, c0=C0, ppw=12,
    )
    grid.print_info()

    tips = create_tips_transducer(grid, n_layers=3)
    n_sources = tips.source_coords_px.shape[0]
    print(f"TIPS: {n_sources} source pixels, {tips.n_elements} elements")

    # Water medium
    bg = BackgroundDomain(grid=grid, background_property_name='water')
    mb = MediumBuilder(grid=grid)
    mb.register_domain(bg)
    medium = mb.run()
    medium.sound_speed[:] = C0
    medium.density[:] = RHO0
    medium_exp = medium.build_exponential()

    # Focusing delays
    focal_x_px = FOCAL_DEPTH / grid.dx
    focal_y_px = grid.ny / 2.0
    focal_z_px = grid.nz / 2.0
    src_x = tips.source_coords_px[:, 0].astype(np.float64)
    src_y = tips.source_coords_px[:, 1].astype(np.float64)
    src_z = tips.source_coords_px[:, 2].astype(np.float64)
    dist_px = np.sqrt(
        (src_x - focal_x_px)**2 + (src_y - focal_y_px)**2 + (src_z - focal_z_px)**2)
    delays = dist_px * grid.dx / C0
    for elem in range(1, tips.n_elements + 1):
        mask = tips.element_ids == elem
        if mask.any():
            delays[mask] = delays[mask].mean()
    delays = delays.max() - delays

    # Source signals
    omega0 = 2.0 * np.pi * F0
    coeff = 1.05 / (NCYCLES * np.pi)
    worst_pulse = fullwave.utils.pulse.gaussian_modulated_sinusoidal_signal(
        nt=grid.nt, f0=F0, duration=duration,
        ncycles=NCYCLES, drop_off=DUR, p0=P0,
        i_layer=0, dt_for_layer_delay=grid.dt,
        cfl_for_layer_delay=grid.cfl, delay_sec=delays.max())
    threshold = np.abs(worst_pulse).max() * 1e-6
    active_idx = np.nonzero(np.abs(worst_pulse) > threshold)[0]
    n_tic = active_idx[-1] + 1 if len(active_idx) > 0 else grid.nt

    dt_val = duration / grid.nt
    t_base = np.arange(n_tic, dtype=np.float64) * dt_val
    p0_signals = np.zeros((n_sources, n_tic), dtype=np.float32)
    t_offsets = NCYCLES / F0 + delays
    for ch_start in range(0, n_sources, 50000):
        ch_end = min(ch_start + 50000, n_sources)
        t_shifted = t_base[np.newaxis, :] - t_offsets[ch_start:ch_end, np.newaxis]
        w_t = t_shifted * omega0
        a_sq = (coeff * w_t) ** 2
        env = np.exp(-a_sq * a_sq)
        p0_signals[ch_start:ch_end, :] = (env * np.sin(w_t) * P0).astype(np.float32)

    source = fullwave.Source(p0_signals, tips.source_mask)

    mod_x, mod_y, mod_z = 4, 4, 4
    sampling_modulus_time = 7
    sensor = fullwave.Sensor(
        mod_x=mod_x, mod_y=mod_y, mod_z=mod_z,
        sampling_modulus_time=sampling_modulus_time)

    fw_solver = fullwave.Solver(
        work_dir=Path(OUTDIR), grid=grid, medium=medium_exp,
        source=source, sensor=sensor,
        run_on_memory=False, use_exponential_attenuation=True)

    t0 = _time.time()
    fw_solver.run(load_results=False)
    elapsed = _time.time() - t0
    print(f"FDTD completed in {elapsed:.1f} s")

    # Load sparse-grid output via memmap
    sim_dir = Path(OUTDIR) / 'txrx_0'
    grid_dat = sim_dir / 'genout_grid.dat'
    print(f"Loading results from {grid_dat}")
    raw = np.memmap(str(grid_dat), dtype=np.float32, mode='r')

    nx_ds = int(np.ceil(grid.nx / mod_x))
    ny_ds = int(np.ceil(grid.ny / mod_y))
    nz_ds = int(np.ceil(grid.nz / mod_z))
    n_sensors = nx_ds * ny_ds * nz_ds
    nt_rec = len(raw) // n_sensors

    ppp = np.full(n_sensors, -np.inf, dtype=np.float32)
    pnp = np.full(n_sensors, np.inf, dtype=np.float32)
    pI = np.zeros(n_sensors, dtype=np.float64)
    for t in range(nt_rec):
        p = raw[t * n_sensors : (t+1) * n_sensors]
        ppp = np.maximum(ppp, p)
        pnp = np.minimum(pnp, p)
        pI += p.astype(np.float64)**2

    ppp = ppp.reshape(nx_ds, ny_ds, nz_ds)
    pnp = pnp.reshape(nx_ds, ny_ds, nz_ds)
    pI = pI.reshape(nx_ds, ny_ds, nz_ds)

    # Weight pI by dT so it represents sum(p²)*dT
    fdtd_dt = grid.dt * sampling_modulus_time
    pI = pI * fdtd_dt

    # Depth axis: parse sensor body start from FDTD execution log
    dx_ds = grid.dx * mod_x
    pml_px = 24
    sensor_start_px = pml_px + 8  # default: PML + transition
    log_path = Path(OUTDIR) / 'txrx_0' / 'fw2_execution.log'
    if log_path.exists():
        import re
        log_text = log_path.read_text()
        m = re.search(r'body \[(\d+),', log_text)
        if m:
            sensor_start_px = int(m.group(1))
            print(f"  Sensor body start from log: {sensor_start_px}")
    xaxis = (np.arange(nx_ds) * mod_x + sensor_start_px - pml_px) * grid.dx
    yaxis = (np.arange(ny_ds) - ny_ds // 2) * dx_ds

    np.savez(os.path.join(OUTDIR, 'fdtd_results.npz'),
             ppp=ppp, pnp=pnp, pI=pI,
             xaxis=xaxis, yaxis=yaxis, dx_ds=dx_ds,
             dx=grid.dx, fdtd_dt=fdtd_dt)
    return dict(ppp=ppp, pnp=pnp, pI=pI, xaxis=xaxis, yaxis=yaxis, dx_ds=dx_ds)


# ======================================================================
# ASM
# ======================================================================
def run_asm(mode='planes'):
    """Run ASM simulation of TIPS in water."""
    print(f'\n--- ASM ({mode}): TIPS in water ---')

    lam = C0 / F0
    dX = lam / 5          # finer grid for accuracy (was lam/4)
    domain_lat = 112e-3
    nX = int(np.ceil(domain_lat / dX))
    if nX % 2 == 0:
        nX += 1
    nY = nX

    delay_spread_est = (np.sqrt(ROC**2 + OUTER_R**2) - ROC) / C0
    pulse_duration = 2 * NCYCLES / F0 + delay_spread_est + 3e-6
    dT = dX / (5 * C0)
    nT = int(np.ceil(pulse_duration / dT))
    if nT % 2 == 0:
        nT += 1
    nT = min(nT, 601)

    xaxis = np.arange(nX) * dX - (nX - 1) * dX / 2
    yaxis = np.arange(nY) * dX - (nY - 1) * dX / 2
    taxis = np.arange(nT) * dT - (nT - 1) * dT / 2

    dZ = dX              # finer propagation step (was 2*dX)
    prop_dist = 80e-3

    print(f'  Grid: {nX}x{nY}x{nT}, dX={dX*1e3:.3f} mm, '
          f'dT={dT*1e9:.2f} ns, dZ={dZ*1e3:.3f} mm')

    if mode == 'flat':
        initial_field = make_bowl_source(
            xaxis, yaxis, taxis, F0, C0, P0,
            radius=OUTER_R, roc=ROC, focus=FOCAL_DEPTH,
            ncycles=NCYCLES, dur=DUR,
            inner_radius=INNER_R, n_elements=N_ELEMENTS)
        sp = None
    else:
        source_planes_list, bowl_depth = make_bowl_source_planes(
            xaxis, yaxis, taxis, F0, C0, P0,
            radius=OUTER_R, roc=ROC, dZ=dZ, focus=FOCAL_DEPTH,
            ncycles=NCYCLES, dur=DUR,
            inner_radius=INNER_R, n_elements=N_ELEMENTS)
        print(f'  Bowl depth: {bowl_depth*1e3:.2f} mm, '
              f'{len(source_planes_list)} z-slices')
        initial_field = source_planes_list[0][1]
        sp = source_planes_list[1:]

    params = SolverParams(
        dX=dX, dY=dX, dT=dT,
        c0=C0, rho0=RHO0,
        beta=0.0,
        alpha0=-1.0,
        f0=F0,
        propDist=prop_dist,
        boundaryFactor=0.15,
        useSplitStep=True,
        useAdaptiveFiltering=True,
        useTVD=False,
        dZmin=dZ,
        boundaryProfile='wendland',
        useFreqWeightedBoundary=True,
        useSuperAbsorbing=True,
        superAbsorbingStrength=0.5,
        sourcePlanes=sp,
    )

    t0 = _time.time()
    field, pnp, ppp, pI, pIloss, zaxis, pax = angular_spectrum_solve(
        initial_field, params, verbose=True)
    elapsed = _time.time() - t0
    print(f'ASM ({mode}) completed in {elapsed:.1f} s')

    # Weight pI by dT so it represents sum(p²)*dT (comparable to FDTD)
    pI = pI * dT

    np.savez(os.path.join(OUTDIR, f'asm_{mode}_results.npz'),
             ppp=ppp, pnp=pnp, pI=pI, zaxis=zaxis,
             xaxis=xaxis, yaxis=yaxis, dX=dX,
             initial_field_peak=np.max(np.abs(initial_field)),
             field=field)

    return dict(ppp=ppp, pnp=pnp, pI=pI, zaxis=zaxis,
                xaxis=xaxis, yaxis=yaxis, dX=dX)


# ======================================================================
# Figure A: TIPS setup overview (matching Figure 14 style)
# ======================================================================
def make_setup_figure(asm_planes):
    """Setup overview: bowl geometry, source slices, aperture, delays."""
    print('\n--- Generating setup figure ---')

    lam = C0 / F0
    dX = lam / 5
    nX = len(asm_planes['xaxis'])
    nY = nX
    xaxis = asm_planes['xaxis']
    yaxis = asm_planes['yaxis']
    dZ = dX

    # Recompute source slices for visualization
    dT = dX / (5 * C0)
    delay_spread_est = (np.sqrt(ROC**2 + OUTER_R**2) - ROC) / C0
    pulse_duration = 2 * NCYCLES / F0 + delay_spread_est + 3e-6
    nT = int(np.ceil(pulse_duration / dT))
    if nT % 2 == 0:
        nT += 1
    nT = min(nT, 601)
    taxis = np.arange(nT) * dT - (nT - 1) * dT / 2

    source_planes_list, bowl_depth = make_bowl_source_planes(
        xaxis, yaxis, taxis, F0, C0, P0,
        radius=OUTER_R, roc=ROC, dZ=dZ, focus=FOCAL_DEPTH,
        ncycles=NCYCLES, dur=DUR,
        inner_radius=INNER_R, n_elements=N_ELEMENTS)

    ext_xy = [xaxis[0]*1e3, xaxis[-1]*1e3, yaxis[0]*1e3, yaxis[-1]*1e3]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)

    # --- Top left: Bowl cross-section schematic ---
    ax = axes[0, 0]
    r_vals = np.linspace(-OUTER_R*1e3, OUTER_R*1e3, 500)
    r_m = r_vals * 1e-3
    z_bowl = ROC - np.sqrt(np.maximum(ROC**2 - r_m**2, 0))
    z_bowl_mm = z_bowl * 1e3
    bowl_mask = (np.abs(r_m) >= INNER_R) & (np.abs(r_m) <= OUTER_R)
    z_plot = np.where(bowl_mask, z_bowl_mm, np.nan)
    ax.plot(r_vals, z_plot, 'b-', lw=2.5, label='Bowl surface')
    dr = (OUTER_R - INNER_R) / N_ELEMENTS
    for ie in range(N_ELEMENTS + 1):
        r_bound = (INNER_R + ie * dr) * 1e3
        z_bound = (ROC - np.sqrt(ROC**2 - (INNER_R + ie * dr)**2)) * 1e3
        ax.plot([r_bound, r_bound], [0, z_bound], 'b:', lw=0.5, alpha=0.5)
        ax.plot([-r_bound, -r_bound], [0, z_bound], 'b:', lw=0.5, alpha=0.5)
    ax.plot(0, FOCAL_DEPTH*1e3, 'r*', ms=12, label=f'Focus ({FOCAL_DEPTH*1e3:.0f} mm)')
    for sign in [-1, 1]:
        ax.plot([sign*OUTER_R*1e3, 0],
                [z_bowl_mm[bowl_mask][-1 if sign > 0 else 0], FOCAL_DEPTH*1e3],
                'r--', lw=0.8, alpha=0.5)
    ax.set(xlabel='r (mm)', ylabel='z (mm)',
           title=f'Bowl cross-section')
    ax.set_ylim([-2, 60])
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.grid(True)

    # --- Top right: Plane-by-plane source slices ---
    ax = axes[0, 1]
    for z_pos, sp_field in source_planes_list:
        sp_mag = np.max(np.abs(sp_field), axis=2)
        lat_profile = sp_mag[:, nY//2]
        nonzero = lat_profile > 0
        if nonzero.any():
            r_min_mm = xaxis[nonzero][0] * 1e3
            r_max_mm = xaxis[nonzero][-1] * 1e3
            ax.barh(z_pos*1e3, r_max_mm - r_min_mm,
                    left=r_min_mm, height=dZ*1e3*0.8,
                    color='C0', alpha=0.6, edgecolor='navy', lw=0.5)
            ax.barh(z_pos*1e3, -(r_max_mm - r_min_mm),
                    left=-r_min_mm, height=dZ*1e3*0.8,
                    color='C0', alpha=0.6, edgecolor='navy', lw=0.5)
    ax.plot(0, FOCAL_DEPTH*1e3, 'r*', ms=12)
    ax.set(xlabel='r (mm)', ylabel='z (mm)',
           title=f'Source slices ({len(source_planes_list)} layers)')
    ax.set_ylim([-2, 20])
    ax.invert_yaxis()
    ax.grid(True)

    # --- Bottom left: Annular aperture ---
    ax = axes[1, 0]
    XX, YY = np.meshgrid(xaxis, yaxis, indexing='ij')
    r = np.sqrt(XX**2 + YY**2)
    aperture = ((r <= OUTER_R) & (r >= INNER_R)).astype(float)
    elem_map = np.zeros_like(aperture)
    for ie in range(N_ELEMENTS):
        r_lo = INNER_R + ie * dr
        r_hi = INNER_R + (ie + 1) * dr
        mask = (r >= r_lo) & (r < r_hi) & (aperture > 0)
        elem_map[mask] = ie + 1
    im = ax.imshow(elem_map.T, extent=ext_xy, aspect='equal',
                   cmap='YlOrRd', vmin=0, vmax=8)
    ax.set(xlabel='x (mm)', ylabel='y (mm)',
           title=f'Aperture ({N_ELEMENTS} elements)')
    plt.colorbar(im, ax=ax, label='Element ID')

    # --- Bottom right: Focusing delay map ---
    ax = axes[1, 1]
    r2 = XX**2 + YY**2
    r2_clipped = np.minimum(r2, ROC**2 - 1e-20)
    z_bowl_2d = ROC - np.sqrt(ROC**2 - r2_clipped)
    dist_to_focus = np.sqrt(r2 + (FOCAL_DEPTH - z_bowl_2d)**2)
    delays = np.zeros_like(r)
    ap_mask = aperture > 0
    delays[ap_mask] = dist_to_focus[ap_mask] / C0
    for ie in range(N_ELEMENTS):
        r_lo = INNER_R + ie * dr
        r_hi = INNER_R + (ie + 1) * dr
        emask = (r >= r_lo) & (r < r_hi) & ap_mask
        if emask.any():
            delays[emask] = delays[emask].mean()
    delays[ap_mask] = delays[ap_mask].max() - delays[ap_mask]
    delays[~ap_mask] = np.nan
    im = ax.imshow(delays.T * 1e6, extent=ext_xy, aspect='equal', cmap='viridis')
    ax.set(xlabel='x (mm)', ylabel='y (mm)',
           title=f'Focusing delays')
    plt.colorbar(im, ax=ax, label=r'$\mu$s')

    fig.suptitle(f'TIPS Transducer: f$_0$={F0/1e6:.0f} MHz, '
                 f'ROC={ROC*1e3:.0f} mm, focus={FOCAL_DEPTH*1e3:.0f} mm',
                 fontsize=14)
    _save_fig(fig, 'tips_setup.png')


# ======================================================================
# Figure B: Three-way comparison (matching Figure 15 style)
# ======================================================================
def make_comparison_figure(fdtd, asm_planes):
    """ASM plane-by-plane vs FDTD comparison — dB scale."""
    print('\n--- Generating comparison figure ---')

    dB_floor = -40

    # --- FDTD grid ---
    z_fdtd = fdtd['xaxis']
    y_fdtd = fdtd['yaxis']
    mid_y_f = len(y_fdtd) // 2
    mid_z_f = mid_y_f

    pI_fdtd = fdtd['pI']
    onaxis_fdtd = pI_fdtd[:, mid_y_f, mid_z_f]
    focal_idx_fdtd = np.argmax(onaxis_fdtd)
    focal_z_fdtd = z_fdtd[focal_idx_fdtd]
    print(f'  FDTD focal depth: {focal_z_fdtd*1e3:.1f} mm')

    # --- ASM data ---
    xaxis = asm_planes['xaxis']
    yaxis = asm_planes['yaxis']
    nX = len(xaxis)
    nY = len(yaxis)
    cx = nX // 2
    cy = nY // 2

    onaxis_asm = asm_planes['pI'][cx, cy, :]
    far_mask = asm_planes['zaxis'] > 20e-3
    far_start = np.argmax(far_mask)
    focal_idx_asm = far_start + np.argmax(onaxis_asm[far_start:])
    focal_z_asm = asm_planes['zaxis'][focal_idx_asm]
    print(f'  ASM focal depth: {focal_z_asm*1e3:.1f} mm')

    # --- Interpolate FDTD onto ASM grids ---
    z_asm = asm_planes['zaxis']
    z_common = z_asm[z_asm <= z_fdtd[-1]]
    n_zc = len(z_common)

    interp_onaxis = RegularGridInterpolator(
        (z_fdtd,), onaxis_fdtd, method='linear',
        bounds_error=False, fill_value=0)
    fdtd_onaxis_interp = interp_onaxis(z_common[:, None]).ravel()

    fdtd_xz = pI_fdtd[:, :, mid_z_f]
    interp_xz = RegularGridInterpolator(
        (z_fdtd, y_fdtd), fdtd_xz,
        method='linear', bounds_error=False, fill_value=0)
    z_pts, x_pts = np.meshgrid(z_common, xaxis, indexing='ij')
    fdtd_xz_interp = interp_xz(
        np.stack([z_pts.ravel(), x_pts.ravel()], axis=-1)
    ).reshape(n_zc, nX)

    fdtd_yz = pI_fdtd[:, mid_y_f, :]
    interp_yz = RegularGridInterpolator(
        (z_fdtd, y_fdtd), fdtd_yz,
        method='linear', bounds_error=False, fill_value=0)
    z_pts_y, y_pts = np.meshgrid(z_common, yaxis, indexing='ij')
    fdtd_yz_interp = interp_yz(
        np.stack([z_pts_y.ravel(), y_pts.ravel()], axis=-1)
    ).reshape(n_zc, nY)

    fdtd_focal_xy = pI_fdtd[focal_idx_fdtd, :, :]
    interp_focal = RegularGridInterpolator(
        (y_fdtd, y_fdtd), fdtd_focal_xy,
        method='linear', bounds_error=False, fill_value=0)
    xy_pts = np.meshgrid(xaxis, yaxis, indexing='ij')
    pts = np.stack([xy_pts[0].ravel(), xy_pts[1].ravel()], axis=-1)
    fdtd_focal_on_asm = interp_focal(pts).reshape(nX, nY)

    fdtd_focal_lat = fdtd_xz_interp[np.argmin(np.abs(z_common - focal_z_fdtd)), :]
    fdtd_focal_elev = fdtd_yz_interp[np.argmin(np.abs(z_common - focal_z_fdtd)), :]

    # --- Normalize: each 2D map self-normalized, line plots on common scale ---
    peak_asm = np.max(asm_planes['pI'])
    peak_fdtd = np.max(pI_fdtd)
    print(f'  ASM peak: {peak_asm:.3e}')
    print(f'  FDTD peak: {peak_fdtd:.3e}')
    print(f'  Ratio (FDTD/ASM): {peak_fdtd/peak_asm:.2f} '
          f'= {10*np.log10(peak_fdtd/peak_asm):.1f} dB')

    def to_dB_self(arr, peak):
        return np.maximum(10 * np.log10(np.maximum(arr, 0) / peak + 1e-30), dB_floor)

    # ===== Figure: 3×2 (ASM planes | FDTD) =====
    fig, axes = plt.subplots(3, 2, figsize=(10, 12.5), constrained_layout=True)

    ext_xz = [xaxis[0]*1e3, xaxis[-1]*1e3, z_common[-1]*1e3, 0]
    ext_lat = [xaxis[0]*1e3, xaxis[-1]*1e3, yaxis[0]*1e3, yaxis[-1]*1e3]

    # --- Row 0: x-z beam cross-sections (each self-normalized) ---
    asm_xz = asm_planes['pI'][:, cy, :n_zc].T
    for col, (title, dat, pk) in enumerate([
        ('ASM plane-by-plane', asm_xz, peak_asm),
        ('FDTD', fdtd_xz_interp, peak_fdtd),
    ]):
        ax = axes[0, col]
        im = ax.imshow(to_dB_self(dat, pk), aspect='auto', extent=ext_xz,
                       origin='upper', vmin=dB_floor, vmax=0, cmap='inferno')
        ax.set(xlabel='x (mm)', ylabel='Depth (mm)', title=f'{title} x-z')

    # --- Row 1: focal x-y planes (each self-normalized) ---
    for col, (title, dat, fz, pk) in enumerate([
        ('ASM', asm_planes['pI'][:, :, focal_idx_asm], focal_z_asm, peak_asm),
        ('FDTD', fdtd_focal_on_asm, focal_z_fdtd, peak_fdtd),
    ]):
        ax = axes[1, col]
        im_mid = ax.imshow(to_dB_self(dat, pk).T, extent=ext_lat, aspect='equal',
                           vmin=dB_floor, vmax=0, cmap='inferno')
        ax.set(xlabel='x (mm)', ylabel='y (mm)',
               title=f'{title} x-y (z={fz*1e3:.1f} mm)')

    # --- Row 2: line profiles (each self-normalized for shape comparison) ---
    iz_asm = np.argmin(np.abs(asm_planes['zaxis'] - focal_z_fdtd))

    # On-axis
    ax = axes[2, 0]
    ax.plot(asm_planes['zaxis']*1e3, to_dB_self(onaxis_asm, peak_asm),
            'r-', lw=2, label='ASM')
    ax.plot(z_common*1e3, to_dB_self(fdtd_onaxis_interp, peak_fdtd),
            'k--', lw=2, label='FDTD')
    ax.axvline(FOCAL_DEPTH*1e3, color='gray', ls=':', alpha=0.5,
               label=f'Focus ({FOCAL_DEPTH*1e3:.0f} mm)')
    ax.set(xlabel='Depth (mm)', ylabel='Intensity (dB)',
           title='On-axis intensity', ylim=[dB_floor, 5])
    ax.legend(fontsize=9)
    ax.grid(True)

    # Lateral (x) and elevational (y) at FDTD focal depth
    ax = axes[2, 1]
    ax.plot(xaxis*1e3, to_dB_self(asm_planes['pI'][:, cy, iz_asm], peak_asm),
            'r-', lw=2, label='ASM lateral')
    ax.plot(xaxis*1e3, to_dB_self(fdtd_focal_lat, peak_fdtd),
            'k--', lw=2, label='FDTD lateral')
    ax.plot(yaxis*1e3, to_dB_self(asm_planes['pI'][cx, :, iz_asm], peak_asm),
            'r:', lw=1.5, label='ASM elevational')
    ax.plot(yaxis*1e3, to_dB_self(fdtd_focal_elev, peak_fdtd),
            'k:', lw=1.5, label='FDTD elevational')
    ax.set(xlabel='Position (mm)', ylabel='Intensity (dB)',
           title=f'Lateral & elevational at z={focal_z_fdtd*1e3:.1f} mm',
           ylim=[dB_floor, 5])
    ax.legend(fontsize=8)
    ax.grid(True)

    fig.colorbar(im, ax=axes[0, :], shrink=0.85, label='dB')
    fig.colorbar(im_mid, ax=axes[1, :], shrink=0.85, label='dB')

    fig.suptitle(f'TIPS Bowl: ASM vs FDTD (water, f$_0$={F0/1e6:.0f} MHz, '
                 f'focus={FOCAL_DEPTH*1e3:.0f} mm)', fontsize=14)
    _save_fig(fig, 'tips_comparison.png')

    # --- Print summary ---
    methods = ['ASM planes', 'FDTD']
    z_focals = [focal_z_asm*1e3, focal_z_fdtd*1e3]
    p_peaks = [
        asm_planes['ppp'][cx, cy, :].max()/1e6,
        fdtd['ppp'][:, mid_y_f, mid_z_f].max()/1e6,
    ]

    print(f'\n{"="*55}')
    print(f'TIPS Bowl Validation Summary (water)')
    print(f'{"="*55}')
    print(f'{"Method":<16} {"Focal (mm)":>10} {"Peak (MPa)":>11} {"Depth err":>10}')
    print(f'{"-"*55}')
    for m, z, p in zip(methods, z_focals, p_peaks):
        err = abs(z - z_focals[1]) / z_focals[1] * 100
        print(f'{m:<16} {z:>10.1f} {p:>11.2f} {err:>9.1f}%')

    focal_loss_dB = 10 * np.log10(
        asm_planes['pI'][cx, cy, focal_idx_asm] /
        (fdtd['pI'][:, mid_y_f, mid_z_f].max() + 1e-30) + 1e-30)
    print(f'\nASM focal intensity vs FDTD: {focal_loss_dB:+.1f} dB')


# ======================================================================
if __name__ == '__main__':
    print('TIPS Bowl Transducer Validation: ASM vs FDTD (water)')
    print('=' * 60)

    # FDTD — check for cached results
    fdtd_cache = os.path.join(OUTDIR, 'fdtd_results.npz')
    if os.path.exists(fdtd_cache):
        print(f'Loading cached FDTD results from {fdtd_cache}')
        d = np.load(fdtd_cache)
        fdtd = {k: d[k] for k in d.files}
    else:
        fdtd = run_fdtd()

    # ASM — check for cached results
    planes_cache = os.path.join(OUTDIR, 'asm_planes_results.npz')

    if os.path.exists(planes_cache):
        print(f'Loading cached ASM planes results')
        d = np.load(planes_cache)
        asm_planes = {k: d[k] for k in d.files}
    else:
        asm_planes = run_asm(mode='planes')

    make_setup_figure(asm_planes)
    make_comparison_figure(fdtd, asm_planes)
