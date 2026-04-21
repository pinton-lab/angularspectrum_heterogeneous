"""
Transcranial benchmark: Angular spectrum + phase screens vs FDTD reference.

Uses the Halle skull microCT and sparse 1024-element array geometry from
../shearwave/examples/sparse_transcranial_shear_3d.py.

The skull CT is converted into a stack of phase screens at each propagation
step.  The ASM propagates the focused beam through these screens and the
resulting focal field is compared against FDTD results (when available).

If no FDTD data exists, the script produces the ASM-only transcranial
simulation and saves diagnostic figures.
"""

import os, sys
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from angular_spectrum_solver import (
    SolverParams, angular_spectrum_solve, generate_phase_screen,
    precalculate_mas, ablvec_wendland
)

OUTDIR = os.path.join(os.path.dirname(__file__), 'validation_results', 'transcranial')
os.makedirs(OUTDIR, exist_ok=True)

NRRD_PATH = '/celerina/gfp/mfs/fullwave2_sparse_transcranial/skull_microCT_zenodo/halle_skull.nrrd'
CONNECTOR_PATH = '/celerina/gfp/mfs/fullwave2_sparse_transcranial/Sparse_TransConnector.mat'
FDTD_OUTPUT_DIR = '/home/gfp/fullwave25-private/outputs/sparse_transcranial_shear'


def _save_fig(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def load_skull_slab(nrrd_path, placement, slab_size_m, dx_target_m):
    """Extract a rotated skull slab aligned with the beam direction.

    Uses the Placement normal/tangent vectors to sample the microCT
    in beam-aligned coordinates, producing a slab where the skull
    appears flat (perpendicular to propagation axis).

    Parameters
    ----------
    nrrd_path : str — path to the NRRD file
    placement : dict — placement parameters with keys:
        xdc_center_lps: (L, P, S) in mm
        normal: (nL, nP) inward beam direction
        tangent: (tL, tP) along transducer face
        z_mm: superior coordinate of beam center
    slab_size_m : tuple — (lateral, elevational, depth) in meters
    dx_target_m : float — target grid spacing in meters

    Returns
    -------
    c_map : ndarray (nLat, nElev, nDepth) — sound speed map (m/s)
    rho_map : ndarray — density map (kg/m³)
    dz_slab : float — depth spacing in meters
    """
    import nrrd
    from scipy.interpolate import RegularGridInterpolator

    print(f'Loading skull CT from {nrrd_path}...')
    data, header = nrrd.read(nrrd_path)
    voxel_mm = float(header['space directions'][0, 0])  # isotropic

    # Build LPS coordinate axes for the NRRD volume (in mm)
    lps_axes = tuple(np.arange(data.shape[dim]) * voxel_mm for dim in range(3))

    interpolator = RegularGridInterpolator(
        lps_axes, data.astype(np.float64),
        method='linear', bounds_error=False, fill_value=np.nan)

    # Placement parameters
    xdc_l, xdc_p, _ = placement['xdc_center_lps']
    norm_l, norm_p = placement['normal']
    tang_l, tang_p = placement['tangent']
    z_center_mm = placement['z_mm']

    w_lat, w_elev, w_depth = slab_size_m

    # Build slab coordinate vectors
    slab_res_m = dx_target_m
    n_lat = int(np.round(w_lat / slab_res_m))
    n_elev = int(np.round(w_elev / slab_res_m))
    n_depth = int(np.round(w_depth / slab_res_m))

    x_lateral = np.linspace(-w_lat / 2, w_lat / 2, n_lat)  # centered
    z_elevation = np.linspace(0, w_elev, n_elev)
    y_depth = np.linspace(0, w_depth, n_depth)

    print(f'  Slab grid: {n_lat} x {n_elev} x {n_depth} '
          f'(lat x elev x depth), res={slab_res_m*1e3:.3f} mm')

    # Convert to mm for LPS sampling
    lat_mm = x_lateral * 1e3
    depth_mm = y_depth * 1e3
    w_elev_mm = w_elev * 1e3
    elev_mm = z_elevation * 1e3

    # Precompute lateral-depth LPS grid (constant across elevation)
    lat_grid, depth_grid = np.meshgrid(lat_mm, depth_mm, indexing='ij')
    l_base = xdc_l + lat_grid * tang_l + depth_grid * norm_l
    p_base = xdc_p + lat_grid * tang_p + depth_grid * norm_p

    # Sample slice-by-slice along elevation
    hu_slab = np.full((n_lat, n_elev, n_depth), np.nan, dtype=np.float64)
    for i_elev in range(n_elev):
        s_val = z_center_mm + (elev_mm[i_elev] - w_elev_mm / 2)
        s_plane = np.full_like(l_base, s_val)
        points = np.stack([l_base.ravel(), p_base.ravel(), s_plane.ravel()], axis=-1)
        hu_slab[:, i_elev, :] = interpolator(points).reshape(n_lat, n_depth)

    n_valid = np.count_nonzero(~np.isnan(hu_slab))
    print(f'  Valid voxels: {n_valid}/{hu_slab.size} '
          f'({100*n_valid/hu_slab.size:.1f}%)')

    # Replace NaN with background (tissue)
    hu_slab = np.nan_to_num(hu_slab, nan=0.0)

    # Map HU to acoustic properties (matching FDTD code)
    hu_lower = 700
    hu_upper = 1973
    frac = np.clip((hu_slab - hu_lower) / (hu_upper - hu_lower), 0, 1)

    c_map = (1540.0 + frac * (2900.0 - 1540.0)).astype(np.float32)
    rho_map = (1000.0 + frac * (2200.0 - 1000.0)).astype(np.float32)

    print(f'  c range: [{c_map.min():.0f}, {c_map.max():.0f}] m/s')
    print(f'  rho range: [{rho_map.min():.0f}, {rho_map.max():.0f}] kg/m³')

    # dz for phase screens = depth spacing
    dz_slab = y_depth[1] - y_depth[0]

    return c_map, rho_map, dz_slab


def load_skull_reference_slice(nrrd_path, s_mm):
    """Load a whole-skull axial reference slice at the requested superior coordinate."""
    import nrrd

    data, header = nrrd.read(nrrd_path)
    voxel_mm = float(header['space directions'][0, 0])  # isotropic

    l_axis_mm = np.arange(data.shape[0]) * voxel_mm
    p_axis_mm = np.arange(data.shape[1]) * voxel_mm
    s_axis_mm = np.arange(data.shape[2]) * voxel_mm
    s_idx = int(np.argmin(np.abs(s_axis_mm - s_mm)))

    hu_slice = np.nan_to_num(data[:, :, s_idx], nan=0.0)
    hu_lower = 700
    hu_upper = 1973
    frac = np.clip((hu_slice - hu_lower) / (hu_upper - hu_lower), 0, 1)
    c_slice = (1540.0 + frac * (2900.0 - 1540.0)).astype(np.float32)

    return l_axis_mm, p_axis_mm, c_slice, s_axis_mm[s_idx]


def skull_to_phase_screens(c_map, dz_screen, dz_prop, c0, f0,
                           rho_map=None, rho0=1000.0,
                           alpha_bone=8.0, alpha_tissue=0.5):
    """Convert a 3D sound-speed/density map into phase+amplitude screens.

    Parameters
    ----------
    c_map : (nx, ny, nz) sound-speed map (m/s)
    dz_screen : float — physical thickness of each screen (m)
    dz_prop : float — propagation step size (m)
    c0 : float — background sound speed (m/s)
    f0 : float — center frequency (Hz)
    rho_map : (nx, ny, nz) density map (kg/m³), optional
    rho0 : float — background density (kg/m³)
    alpha_bone : float — bone attenuation (dB/cm/MHz), default 8.0
    alpha_tissue : float — tissue attenuation (dB/cm/MHz), default 0.5

    Returns
    -------
    screens : list of (z_position, phase_array, amplitude_array) tuples
        phase_array : (nx, ny) phase shift in radians at f0
        amplitude_array : (nx, ny) transmission factor (0 to 1)
    """
    nx, ny, nz = c_map.shape
    omega = 2 * np.pi * f0
    f_MHz = f0 / 1e6
    screens = []

    for iz in range(nz):
        z_pos = iz * dz_screen
        c_slice = c_map[:, :, iz]

        # Phase shift: omega * thickness * (1/c - 1/c0)
        phase = omega * dz_screen * (1.0 / c_slice - 1.0 / c0)

        # --- Amplitude: attenuation loss through this screen ---
        # Bone fraction determines local attenuation
        bone_frac = np.clip((c_slice - 1540.0) / (2900.0 - 1540.0), 0, 1)
        alpha_local = alpha_tissue + bone_frac * (alpha_bone - alpha_tissue)
        # Convert dB/cm/MHz to Np/m: 1 dB = 0.1151 Np, 1 cm = 0.01 m
        alpha_Np_m = alpha_local * f_MHz * 0.1151 / 0.01  # Np/m
        atten = np.exp(-alpha_Np_m * dz_screen)

        # --- Amplitude: impedance mismatch transmission ---
        if rho_map is not None:
            rho_slice = rho_map[:, :, iz]
            Z_local = rho_slice * c_slice
            Z0 = rho0 * c0
            # Transmission coefficient for normal incidence
            T = 2.0 * Z_local / (Z_local + Z0)
            # For bone→tissue and tissue→bone boundaries, apply sqrt
            # since each screen represents a thin layer, not a full interface
            amplitude = atten * np.abs(T)
        else:
            amplitude = atten

        # Only add non-trivial screens
        if np.max(np.abs(phase)) > 1e-6 or np.min(amplitude) < 0.999:
            screens.append((z_pos,
                            phase.astype(np.float32),
                            amplitude.astype(np.float32)))

    print(f'  Generated {len(screens)} screens from {nz} z-slices')
    # Report total attenuation through skull at center
    if screens:
        total_atten = np.ones((nx, ny), dtype=np.float32)
        for _, _, amp in screens:
            total_atten *= amp
        center_loss = -20 * np.log10(total_atten[nx//2, ny//2] + 1e-30)
        min_trans = total_atten.min()
        print(f'  Total attenuation: center={center_loss:.1f} dB, '
              f'max loss={-20*np.log10(min_trans+1e-30):.1f} dB')
    return screens


def screens_to_xz_maps(screens, mid_elev):
    """Stack phase and attenuation screens into x-z maps at mid-elevation."""
    if not screens:
        return None, None

    phase_xz = np.stack([phase[:, mid_elev] for _, phase, _ in screens], axis=0)
    atten_xz_dB = np.stack(
        [-20.0 * np.log10(np.maximum(amp[:, mid_elev], 1e-30)) for _, _, amp in screens],
        axis=0)
    return phase_xz, atten_xz_dB


def load_sparse_connector(mat_path):
    """Load sparse transducer element positions."""
    with h5py.File(mat_path, 'r') as f:
        positions = f['Position'][()].T  # (1024, 3) in mm
        element_ids = f['SparseConnector'][()].ravel().astype(int)
    return positions, element_ids


def build_initial_condition(positions, element_ids, xaxis, yaxis, t,
                            f0, c0, p0, focal_depth, ncycles=2):
    """Build a focused sparse-array initial condition.

    Maps the 1024-element positions onto the ASM grid and applies
    geometric focusing delays.
    """
    nX = len(xaxis)
    nY = len(yaxis)
    nT = len(t)
    dX = xaxis[1] - xaxis[0]
    omega0 = 2 * np.pi * f0
    dur = 2

    # Map element positions to grid
    iy_raw = np.round(positions[:, 0] / dX).astype(int)
    iz_raw = np.round(positions[:, 1] / dX).astype(int)
    iy_centers = iy_raw - int(np.mean(iy_raw)) + nX // 2
    iz_centers = iz_raw - int(np.mean(iz_raw)) + nY // 2

    # Compute focusing delays
    focal_y = nX // 2
    focal_z = nY // 2
    dist = np.sqrt(focal_depth**2 + (iy_centers * dX - focal_y * dX)**2 +
                   (iz_centers * dX - focal_z * dX)**2)
    delays = (dist.max() - dist) / c0

    # Build aperture and time-delay field
    element_size = max(1, round(750e-6 / dX))  # 750 um pitch
    half = element_size // 2

    ap = np.zeros((nX, nY), dtype=np.float32)
    tt = np.zeros((nX, nY), dtype=np.float32)

    for m in range(len(element_ids)):
        iy = iy_centers[m]
        iz = iz_centers[m]
        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                yi = iy + di
                zi = iz + dj
                if 0 <= yi < nX and 0 <= zi < nY:
                    ap[yi, zi] = 1.0
                    tt[yi, zi] = delays[m]

    # Generate space-time field
    t_grid = t[np.newaxis, np.newaxis, :]
    tt_grid = tt[:, :, np.newaxis]
    field = np.exp(-((1.05 * (t_grid - tt_grid) * omega0 / (ncycles * np.pi))**(2*dur))) \
            * np.sin(omega0 * (t_grid - tt_grid)) * p0
    field = ap[:, :, np.newaxis] * field

    n_active = int(np.sum(ap > 0))
    print(f'  Sparse array: {n_active} active grid points, '
          f'delay spread: {(delays.max()-delays.min())*1e6:.1f} us')

    return field.astype(np.float32)


def main():
    print('Transcranial Benchmark: ASM + Phase Screens')
    print('=' * 60)

    # --- Parameters matching the FDTD simulation ---
    f0 = 1e6
    c0 = 1540.0
    rho0 = 1000.0
    lam = c0 / f0  # 1.54 mm
    p0 = 1.5e6     # 1.5 MPa
    focal_depth = 50e-3
    domain_depth = 65e-3
    domain_lat = 66e-3

    # ASM grid — lambda/5 for finer resolution to reduce FDTD comparison error
    dX = lam / 5  # ~0.31 mm
    dT = dX / (5 * c0)
    nX = int(round(domain_lat / dX))
    nY = nX
    if nX % 2 == 0:
        nX += 1
        nY += 1

    # Time axis — enough for the focused pulse plus delay spread
    ncycles = 2
    duration_min = ncycles * 6 / f0
    delay_spread_est = 15e-6
    duration = 2.0 * (duration_min + delay_spread_est)
    nT = int(round(duration / dT))
    if nT % 2 == 0:
        nT += 1
    # Cap nT to keep memory manageable
    nT = min(nT, 501)

    xaxis = np.arange(nX) * dX - (nX - 1) * dX / 2
    yaxis = np.arange(nY) * dX - (nY - 1) * dX / 2
    t = np.arange(nT) * dT
    t -= np.mean(t)

    print(f'\nGrid: {nX}x{nY}x{nT}, dX={dX*1e3:.3f} mm, '
          f'dT={dT*1e9:.2f} ns, duration={duration*1e6:.1f} us')

    dZ = dX  # propagation step — small for accuracy through skull

    # --- Build transducer, skull slab, and setup figure ---
    print(f'\nLoading transducer from {CONNECTOR_PATH}')
    positions, element_ids = load_sparse_connector(CONNECTOR_PATH)

    print('\nBuilding focused initial condition...')
    apa = build_initial_condition(positions, element_ids, xaxis, yaxis, t,
                                  f0, c0, p0, focal_depth, ncycles=ncycles)

    print('\nLoading skull CT with beam-aligned extraction...')
    placement = {
        'xdc_center_lps': (140.6, 118.5, 85.0),
        'normal': (-0.954, -0.298),    # inward beam direction
        'tangent': (-0.298, 0.954),    # along transducer face
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
        print(f'  Resampled to {c_map.shape}')

    print('\nConverting skull to phase+amplitude screens...')
    screens = skull_to_phase_screens(c_map, dz_screen, dZ, c0, f0,
                                     rho_map=rho_map, rho0=rho0,
                                     alpha_bone=8.0, alpha_tissue=0.5)

    n_lat, n_elev, n_depth = c_map.shape
    depth_axis_mm = np.arange(n_depth) * dz_screen * 1e3
    depth_extent = [xaxis[0]*1e3, xaxis[-1]*1e3, depth_axis_mm[-1], 0]
    elev_depth_extent = [yaxis[0]*1e3, yaxis[-1]*1e3, depth_axis_mm[-1], 0]
    mid_lat = n_lat // 2
    mid_elev = n_elev // 2
    phase_xz, atten_xz_dB = screens_to_xz_maps(screens, mid_elev)

    l_axis_mm, p_axis_mm, whole_c_slice, s_slice_mm = load_skull_reference_slice(
        NRRD_PATH, placement['z_mm'])
    # Match the axial-overview convention used by the shearwave paper:
    # P is the horizontal axis, L is the vertical axis (flipped so L=0 is at
    # the top). In this frame the hardcoded normal/tangent lie flat against
    # the skull outer surface.
    whole_extent = [p_axis_mm[0], p_axis_mm[-1], l_axis_mm[-1], l_axis_mm[0]]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), constrained_layout=True)

    im0 = axes[0, 0].imshow(
        whole_c_slice,
        extent=whole_extent,
        origin='upper',
        aspect='equal',
        cmap='bone')
    axes[0, 0].set(xlabel='Posterior (mm)', ylabel='Left (mm)',
                   title=f'Whole skull axial slice (S={s_slice_mm:.1f} mm)')
    xdc_l, xdc_p, _ = placement['xdc_center_lps']
    tang_l, tang_p = placement['tangent']
    norm_l, norm_p = placement['normal']
    lat_half_mm = 0.5 * slab_size_m[0] * 1e3
    depth_mm = slab_size_m[2] * 1e3

    # Axes are (P, L): pass P as x, L as y for all overlays.
    axes[0, 0].plot(
        [xdc_p - lat_half_mm * tang_p, xdc_p + lat_half_mm * tang_p],
        [xdc_l - lat_half_mm * tang_l, xdc_l + lat_half_mm * tang_l],
        'r-', lw=2)
    axes[0, 0].plot(
        [xdc_p, xdc_p + depth_mm * norm_p],
        [xdc_l, xdc_l + depth_mm * norm_l],
        'c--', lw=2)
    axes[0, 0].plot(xdc_p, xdc_l, 'yo', ms=5)
    plt.colorbar(im0, ax=axes[0, 0], label='m/s')

    im1 = axes[0, 1].imshow(
        np.max(np.abs(apa), axis=2).T,
        extent=[xaxis[0]*1e3, xaxis[-1]*1e3, yaxis[0]*1e3, yaxis[-1]*1e3],
        aspect='equal')
    axes[0, 1].set(xlabel='x (mm)', ylabel='y (mm)',
                   title='Sparse-array source plane')
    plt.colorbar(im1, ax=axes[0, 1], label='Pa')

    im2 = axes[0, 2].imshow(
        c_map[mid_lat, :, :].T,
        extent=elev_depth_extent,
        aspect='auto',
        cmap='bone',
        origin='upper')
    axes[0, 2].set(xlabel='y (mm)', ylabel='Depth (mm)',
                   title='Sound speed y-z')
    plt.colorbar(im2, ax=axes[0, 2], label='m/s')

    im3 = axes[1, 0].imshow(
        c_map[:, mid_elev, :].T,
        extent=depth_extent,
        aspect='auto',
        cmap='bone',
        origin='upper')
    axes[1, 0].set(xlabel='x (mm)', ylabel='Depth (mm)',
                   title='Sound speed x-z')
    plt.colorbar(im3, ax=axes[1, 0], label='m/s')

    if phase_xz is not None:
        phase_lim = np.max(np.abs(phase_xz)) + 1e-12
        im4 = axes[1, 1].imshow(
            phase_xz,
            extent=depth_extent,
            aspect='auto',
            cmap='RdBu_r',
            origin='upper',
            vmin=-phase_lim,
            vmax=phase_lim)
        axes[1, 1].set(xlabel='x (mm)', ylabel='Depth (mm)',
                       title='Phase screens x-z')
        plt.colorbar(im4, ax=axes[1, 1], label='rad')

        im5 = axes[1, 2].imshow(
            atten_xz_dB,
            extent=depth_extent,
            aspect='auto',
            cmap='magma',
            origin='upper')
        axes[1, 2].set(xlabel='x (mm)', ylabel='Depth (mm)',
                       title='Amplitude-screen loss x-z')
        plt.colorbar(im5, ax=axes[1, 2], label='dB')
    else:
        for ax in axes[1, 1:]:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No phase screens', ha='center', va='center')

    fig.suptitle('Transcranial Setup: Whole Skull Context and Beam-Aligned Screens', fontsize=14)
    _save_fig(fig, 'setup_overview.png')

    # --- Check for saved results to skip simulation ---
    saved_results = os.path.join(OUTDIR, 'transcranial_results.npz')
    if os.path.exists(saved_results):
        print(f'\nLoading saved ASM results from {saved_results}...')
        data = np.load(saved_results)
        pI = data['pI']
        pI_h = data['pI_h']
        pnp = data['pnp']
        ppp = data['ppp']
        zaxis = data['zaxis']
        focal_loss_dB = float(data['focal_loss_dB'])
        print(f'  pI shape: {pI.shape}, zaxis: {len(zaxis)} steps')
        print(f'  Focal loss: {focal_loss_dB:.1f} dB')
    else:
        # --- Run ASM propagation with phase screens ---
        print(f'\nRunning ASM propagation with {len(screens)} phase screens...')
        params = SolverParams(
            dX=dX, dY=dX, dT=dT, c0=c0, rho0=rho0, beta=3.5,
            alpha0=0.5, attenPow=1, f0=f0, propDist=domain_depth,
            boundaryFactor=0.12, useSplitStep=True,
            useAdaptiveFiltering=True, useTVD=False,
            fluxScheme='kt',
            boundaryProfile='wendland',
            useFreqWeightedBoundary=True,
            useSuperAbsorbing=True, superAbsorbingStrength=0.5,
            stabilityThreshold=0.2, stabilityRecoveryFactor=0.15,
            dZmin=dZ,
            phaseScreens=screens)

        field, pnp, ppp, pI, pIloss, zaxis, pax = angular_spectrum_solve(
            apa, params, verbose=True)

        # --- Also run without skull for comparison ---
        print('\nRunning ASM propagation WITHOUT skull (homogeneous)...')
        params_homo = SolverParams(
            dX=dX, dY=dX, dT=dT, c0=c0, rho0=rho0, beta=3.5,
            alpha0=0.5, attenPow=1, f0=f0, propDist=domain_depth,
            boundaryFactor=0.12, useSplitStep=True,
            useAdaptiveFiltering=True, useTVD=False,
            fluxScheme='kt',
            boundaryProfile='wendland',
            useFreqWeightedBoundary=True,
            useSuperAbsorbing=True, superAbsorbingStrength=0.5,
            stabilityThreshold=0.2, stabilityRecoveryFactor=0.15,
            dZmin=dZ,
            phaseScreens=None)

        field_h, pnp_h, ppp_h, pI_h, _, zaxis_h, pax_h = angular_spectrum_solve(
            apa, params_homo, verbose=True)

    # --- Results figures: three-way comparison (ASM homo, ASM skull, FDTD) ---
    print('\nGenerating comparison figures...')
    zaxis_h = zaxis  # same grid for both runs

    # Find focal depth (skip near-field z < 30 mm)
    far_mask = zaxis > 30e-3
    far_start = np.argmax(far_mask)
    focal_idx_h = far_start + np.argmax(pI_h[nX//2, nY//2, far_start:])
    focal_idx_s = far_start + np.argmax(pI[nX//2, nY//2, far_start:])
    print(f'  Homo focal: z={zaxis[focal_idx_h]*1e3:.1f} mm')
    print(f'  Skull focal: z={zaxis[focal_idx_s]*1e3:.1f} mm')

    # dB conversion helper
    def to_dB(arr, floor_dB=-40):
        arr_pos = np.maximum(arr, 0)
        peak = np.max(arr_pos)
        if peak < 1e-30:
            return np.full_like(arr, floor_dB)
        return np.maximum(10 * np.log10(arr_pos / peak + 1e-30), floor_dB)

    # --- Try to load FDTD for three-way comparison ---
    fdtd_map = os.path.join(FDTD_OUTPUT_DIR, 'propagation_map.npy')
    has_fdtd = os.path.exists(fdtd_map)
    if has_fdtd:
        from scipy.interpolate import RegularGridInterpolator
        print('\n  FDTD data found — loading for three-way comparison...')
        gp = np.load(os.path.join(FDTD_OUTPUT_DIR, 'grid_params.npz'))
        fdtd_dx = float(gp['dx']) * int(gp['mod_x'])
        fdtd_dt = float(np.fromfile(os.path.join(FDTD_OUTPUT_DIR, 'txrx_0', 'dT.dat'),
                                    dtype=np.float32)[0]) * int(gp['sampling_modulus_time'])
        fdtd_time_scale = fdtd_dt / dT
        p_fdtd = np.load(fdtd_map, mmap_mode='r')
        nt_fdtd, nx_fdtd, ny_fdtd, nz_fdtd = p_fdtd.shape
        pI_fdtd = np.zeros((nx_fdtd, ny_fdtd, nz_fdtd), dtype=np.float64)
        chunk = 50
        for i in range(0, nt_fdtd, chunk):
            end = min(i + chunk, nt_fdtd)
            pI_fdtd += np.sum(np.array(p_fdtd[i:end]).astype(np.float64)**2, axis=0)
        pI_fdtd *= fdtd_time_scale

        fdtd_zaxis = np.arange(nx_fdtd) * fdtd_dx
        fdtd_yaxis = (np.arange(ny_fdtd) - ny_fdtd / 2) * fdtd_dx
        fdtd_xaxis = (np.arange(nz_fdtd) - nz_fdtd / 2) * fdtd_dx
        mid_y_fdtd = ny_fdtd // 2
        mid_z_fdtd = nz_fdtd // 2

        # FDTD on-axis
        onaxis_fdtd = pI_fdtd[:, mid_y_fdtd, mid_z_fdtd]
        focal_idx_fdtd = np.argmax(onaxis_fdtd)
        focal_z_fdtd = fdtd_zaxis[focal_idx_fdtd]
        print(f'  FDTD focal depth: {focal_z_fdtd*1e3:.1f} mm')

        # Interpolate FDTD on-axis onto ASM z-grid
        z_common = zaxis[zaxis <= fdtd_zaxis[-1]]
        interp_onaxis = RegularGridInterpolator(
            (fdtd_zaxis,), onaxis_fdtd, method='linear',
            bounds_error=False, fill_value=0)
        fdtd_onaxis_interp = interp_onaxis(z_common[:, None]).ravel()

        # Interpolate FDTD x-z plane onto ASM grid
        fdtd_xz = pI_fdtd[:, :, mid_z_fdtd]  # (nx_prop, ny_fdtd)
        interp_xz = RegularGridInterpolator(
            (fdtd_zaxis, fdtd_yaxis), fdtd_xz,
            method='linear', bounds_error=False, fill_value=0)
        z_pts, x_pts = np.meshgrid(z_common, xaxis, indexing='ij')
        fdtd_xz_interp = interp_xz(
            np.stack([z_pts.ravel(), x_pts.ravel()], axis=-1)
        ).reshape(len(z_common), nX)

        # Interpolate FDTD focal x-y plane onto the ASM grid
        fdtd_focal_xy = pI_fdtd[focal_idx_fdtd, :, :]
        interp_fdtd_xy = RegularGridInterpolator(
            (fdtd_yaxis, fdtd_xaxis), fdtd_focal_xy,
            method='linear', bounds_error=False, fill_value=0)
        asm_pts = np.meshgrid(xaxis, yaxis, indexing='ij')
        pts = np.stack([asm_pts[0].ravel(), asm_pts[1].ravel()], axis=-1)
        fdtd_focal_on_asm = interp_fdtd_xy(pts).reshape(nX, nY)

        fdtd_elev = pI_fdtd[focal_idx_fdtd, :, mid_z_fdtd]
        interp_elev = RegularGridInterpolator(
            (fdtd_yaxis,), fdtd_elev, method='linear',
            bounds_error=False, fill_value=0)
        fdtd_elev_interp = interp_elev(yaxis[:, None]).ravel()
        print(f'  FDTD intensity uses sum_t(p^2) with recorded dt weighting '
              f'(removed artificial {10*np.log10(nt_fdtd):.1f} dB mean/sum offset)')
        print(f'  Applied FDTD time-step correction: '
              f'10log10({fdtd_dt/dT:.3f}) = {10*np.log10(fdtd_time_scale):+.2f} dB')

    # --- Three-way comparison figure on dB scale ---
    fig, axes = plt.subplots(3, 3, figsize=(15, 12.5), constrained_layout=True)
    dB_floor = -40
    ext_xz = [xaxis[0]*1e3, xaxis[-1]*1e3, zaxis[-1]*1e3, 0]
    ext_lat = [xaxis[0]*1e3, xaxis[-1]*1e3, yaxis[0]*1e3, yaxis[-1]*1e3]

    # Normalize all to the homogeneous peak for absolute comparison
    peak_homo = np.max(pI_h)

    homo_xz_dB = np.maximum(10 * np.log10(pI_h[:, nY//2, :].T / peak_homo + 1e-30), dB_floor)
    im_top = axes[0, 0].imshow(homo_xz_dB, aspect='auto', extent=ext_xz,
                               origin='upper', vmin=dB_floor, vmax=0, cmap='inferno')
    axes[0, 0].set(xlabel='x (mm)', ylabel='Depth (mm)', title='ASM homogeneous x-z')

    skull_xz_dB = np.maximum(10 * np.log10(pI[:, nY//2, :].T / peak_homo + 1e-30), dB_floor)
    axes[0, 1].imshow(skull_xz_dB, aspect='auto', extent=ext_xz,
                      origin='upper', vmin=dB_floor, vmax=0, cmap='inferno')
    axes[0, 1].set(xlabel='x (mm)', ylabel='Depth (mm)', title='ASM through skull x-z')

    if has_fdtd:
        fdtd_xz_dB = np.maximum(
            10 * np.log10(fdtd_xz_interp / peak_homo + 1e-30),
            dB_floor)
        ext_fdtd_xz = [xaxis[0]*1e3, xaxis[-1]*1e3, z_common[-1]*1e3, 0]
        axes[0, 2].imshow(fdtd_xz_dB, aspect='auto', extent=ext_fdtd_xz,
                          origin='upper', vmin=dB_floor, vmax=0, cmap='inferno')
        axes[0, 2].set(xlabel='x (mm)', ylabel='Depth (mm)', title='FDTD x-z')
    else:
        axes[0, 2].axis('off')
        axes[0, 2].text(0.5, 0.5, 'No FDTD data', ha='center', va='center')

    homo_focal_dB = np.maximum(10 * np.log10(pI_h[:, :, focal_idx_h] / peak_homo + 1e-30), dB_floor)
    im_mid = axes[1, 0].imshow(homo_focal_dB.T, extent=ext_lat, aspect='equal',
                               vmin=dB_floor, vmax=0, cmap='inferno')
    axes[1, 0].set(xlabel='x (mm)', ylabel='y (mm)',
                   title=f'ASM homogeneous x-y (z={zaxis[focal_idx_h]*1e3:.1f} mm)')

    skull_focal_dB = np.maximum(10 * np.log10(pI[:, :, focal_idx_s] / peak_homo + 1e-30), dB_floor)
    axes[1, 1].imshow(skull_focal_dB.T, extent=ext_lat, aspect='equal',
                      vmin=dB_floor, vmax=0, cmap='inferno')
    axes[1, 1].set(xlabel='x (mm)', ylabel='y (mm)',
                   title=f'ASM through skull x-y (z={zaxis[focal_idx_s]*1e3:.1f} mm)')

    if has_fdtd:
        fdtd_focal_dB = np.maximum(10 * np.log10(fdtd_focal_on_asm / peak_homo + 1e-30), dB_floor)
        axes[1, 2].imshow(fdtd_focal_dB.T, extent=ext_lat, aspect='equal',
                          vmin=dB_floor, vmax=0, cmap='inferno')
        axes[1, 2].set(xlabel='x (mm)', ylabel='y (mm)',
                       title=f'FDTD x-y (z={focal_z_fdtd*1e3:.1f} mm)')
    else:
        axes[1, 2].axis('off')
        axes[1, 2].text(0.5, 0.5, 'No FDTD data', ha='center', va='center')

    homo_onaxis_dB = np.maximum(10 * np.log10(pI_h[nX//2, nY//2, :] / peak_homo + 1e-30), dB_floor)
    skull_onaxis_dB = np.maximum(10 * np.log10(pI[nX//2, nY//2, :] / peak_homo + 1e-30), dB_floor)
    axes[2, 0].plot(zaxis*1e3, homo_onaxis_dB, 'b-', lw=2, label='ASM homo')
    axes[2, 0].plot(zaxis*1e3, skull_onaxis_dB, 'r--', lw=2, label='ASM skull')
    if has_fdtd:
        fdtd_onaxis_dB = np.maximum(
            10 * np.log10(fdtd_onaxis_interp / peak_homo + 1e-30),
            dB_floor)
        axes[2, 0].plot(z_common*1e3, fdtd_onaxis_dB, 'g:', lw=2, label='FDTD')
    axes[2, 0].set(xlabel='Depth (mm)', ylabel='Intensity (dB re ASM homo peak)',
                   title='On-axis intensity', ylim=[dB_floor, 5])
    axes[2, 0].legend(fontsize=9)
    axes[2, 0].grid(True)

    homo_lat = pI_h[:, nY//2, focal_idx_h]
    skull_lat = pI[:, nY//2, focal_idx_s]
    homo_lat_dB = np.maximum(10 * np.log10(homo_lat / peak_homo + 1e-30), dB_floor)
    skull_lat_dB = np.maximum(10 * np.log10(skull_lat / peak_homo + 1e-30), dB_floor)
    axes[2, 1].plot(xaxis*1e3, homo_lat_dB, 'b-', lw=2,
                    label=f'ASM homo (z={zaxis[focal_idx_h]*1e3:.1f})')
    axes[2, 1].plot(xaxis*1e3, skull_lat_dB, 'r--', lw=2,
                    label=f'ASM skull (z={zaxis[focal_idx_s]*1e3:.1f})')
    if has_fdtd:
        fdtd_focal_lat = fdtd_xz_interp[np.argmin(np.abs(z_common - focal_z_fdtd)), :]
        fdtd_lat_dB = np.maximum(
            10 * np.log10(fdtd_focal_lat / peak_homo + 1e-30),
            dB_floor)
        axes[2, 1].plot(xaxis*1e3, fdtd_lat_dB, 'g:', lw=2,
                        label=f'FDTD (z={focal_z_fdtd*1e3:.1f})')
    axes[2, 1].set(xlabel='x (mm)', ylabel='Intensity (dB re ASM homo peak)',
                   title='Lateral profile at focus', ylim=[dB_floor, 5])
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(True)

    homo_elev = pI_h[nX//2, :, focal_idx_h]
    skull_elev = pI[nX//2, :, focal_idx_s]
    homo_elev_dB = np.maximum(10 * np.log10(homo_elev / peak_homo + 1e-30), dB_floor)
    skull_elev_dB = np.maximum(10 * np.log10(skull_elev / peak_homo + 1e-30), dB_floor)
    axes[2, 2].plot(yaxis*1e3, homo_elev_dB, 'b-', lw=2, label='ASM homo')
    axes[2, 2].plot(yaxis*1e3, skull_elev_dB, 'r--', lw=2, label='ASM skull')
    if has_fdtd:
        fdtd_elev_dB = np.maximum(
            10 * np.log10(fdtd_elev_interp / peak_homo + 1e-30),
            dB_floor)
        axes[2, 2].plot(yaxis*1e3, fdtd_elev_dB, 'g:', lw=2, label='FDTD')
    axes[2, 2].set(xlabel='y (mm)', ylabel='Intensity (dB re ASM homo peak)',
                   title='Elevation profile at focus', ylim=[dB_floor, 5])
    axes[2, 2].legend(fontsize=8)
    axes[2, 2].grid(True)

    fig.suptitle(f'Transcranial Comparison (dB): f0={f0/1e6:.0f} MHz, focal={focal_depth*1e3:.0f} mm, '
                 f'p0={p0/1e6:.1f} MPa', fontsize=14)
    fig.colorbar(im_top, ax=axes[0, :], shrink=0.85, label='dB re ASM homo peak')
    fig.colorbar(im_mid, ax=axes[1, :], shrink=0.85, label='dB re ASM homo peak')
    _save_fig(fig, 'transcranial_comparison.png')

    # Save results if freshly computed — compare peak intensity in focal region
    far_mask_save = zaxis > 30e-3
    skull_focal_peak = np.max(pI[:, :, far_mask_save])
    homo_focal_peak = np.max(pI_h[:, :, far_mask_save])
    focal_loss_dB = 10 * np.log10(skull_focal_peak / (homo_focal_peak + 1e-30) + 1e-30)
    if not os.path.exists(saved_results):
        np.savez(os.path.join(OUTDIR, 'transcranial_results.npz'),
                 pI=pI, pI_h=pI_h, pnp=pnp, ppp=ppp,
                 xaxis=xaxis, yaxis=yaxis, zaxis=zaxis,
                 focal_loss_dB=focal_loss_dB)
    print(f'\n  Focal intensity loss through skull: {focal_loss_dB:.1f} dB')

    # --- Save comparison metrics ---
    if has_fdtd:
        asm_focal_xy = pI[:, :, focal_idx_s]
        fdtd_norm = fdtd_focal_on_asm / (np.max(fdtd_focal_on_asm) + 1e-30)
        asm_norm = asm_focal_xy / (np.max(asm_focal_xy) + 1e-30)
        diff = np.abs(fdtd_norm - asm_norm)
        focal_z_asm = zaxis[focal_idx_s]

        asm_xz_n = pI[:, nY//2, :len(z_common)].T
        asm_xz_n = asm_xz_n / (np.max(asm_xz_n) + 1e-30)
        fdtd_xz_n = fdtd_xz_interp / (np.max(fdtd_xz_interp) + 1e-30)
        xz_diff = np.abs(fdtd_xz_n - asm_xz_n)

        np.savez(os.path.join(OUTDIR, 'fdtd_comparison.npz'),
                 fdtd_focal_norm=fdtd_norm, asm_focal_norm=asm_norm,
                 z_common=z_common,
                 focal_rms_diff=np.sqrt(np.mean(diff**2)),
                 focal_z_fdtd=focal_z_fdtd, focal_z_asm=focal_z_asm)
        print(f'  Focal plane RMS difference: {np.sqrt(np.mean(diff**2)):.4f}')
        print(f'  XZ plane RMS difference: {np.sqrt(np.mean(xz_diff**2)):.4f}')
    else:
        print(f'\nNo FDTD data at {fdtd_map}')
        print('Run the FDTD simulation first:')
        print(f'  python {os.path.abspath("../shearwave/examples/sparse_transcranial_shear_3d.py")} \\')
        print(f'    --nrrd-path {NRRD_PATH} --connector-path {CONNECTOR_PATH}')

    print('\n' + '=' * 60)
    print('Transcranial benchmark complete.')
    print(f'Results saved to {OUTDIR}/')


if __name__ == '__main__':
    main()
