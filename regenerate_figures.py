"""Regenerate transcranial manuscript figures with depth on y-axis, zero at top."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

OUTDIR = 'validation_results/transcranial'
FDTD_OUTPUT_DIR = '/home/gfp/fullwave25-private/outputs/sparse_transcranial_shear'

def to_dB(x, floor=-40):
    xn = x / (np.max(x) + 1e-30)
    return np.clip(10*np.log10(xn + 1e-30), floor, 0)

# --- Load ASM data ---
d = np.load(os.path.join(OUTDIR, 'transcranial_results.npz'))
pI = d['pI']       # (nX, nY, nZ) — skull
pI_h = d['pI_h']   # (nX, nY, nZ) — homogeneous
xaxis = d['xaxis']
yaxis = d['yaxis']
zaxis = d['zaxis']
nX, nY, nZ = pI.shape
mid_x, mid_y = nX//2, nY//2

# Focal indices (skip z < 30 mm)
far_start = np.argmax(zaxis > 30e-3)
focal_idx_s = far_start + np.argmax(pI[mid_x, mid_y, far_start:])
focal_idx_h = far_start + np.argmax(pI_h[mid_x, mid_y, far_start:])
print(f'ASM skull focal: z={zaxis[focal_idx_s]*1e3:.1f} mm')
print(f'ASM homo focal:  z={zaxis[focal_idx_h]*1e3:.1f} mm')

# --- Load FDTD data ---
gp = np.load(os.path.join(FDTD_OUTPUT_DIR, 'grid_params.npz'))
fdtd_dx = float(gp['dx']) * int(gp['mod_x'])
p_fdtd = np.load(os.path.join(FDTD_OUTPUT_DIR, 'propagation_map.npy'), mmap_mode='r')
nt_fdtd, nx_fdtd, ny_fdtd, nz_fdtd = p_fdtd.shape

# FDTD intensity
pI_fdtd = np.zeros((nx_fdtd, ny_fdtd, nz_fdtd), dtype=np.float64)
chunk = 50
for i in range(0, nt_fdtd, chunk):
    end = min(i + chunk, nt_fdtd)
    pI_fdtd += np.sum(np.array(p_fdtd[i:end]).astype(np.float64)**2, axis=0)
pI_fdtd /= nt_fdtd

fdtd_zaxis = np.arange(nx_fdtd) * fdtd_dx
fdtd_yaxis = (np.arange(ny_fdtd) - ny_fdtd / 2) * fdtd_dx
fdtd_xaxis = (np.arange(nz_fdtd) - nz_fdtd / 2) * fdtd_dx

mid_y_fdtd = ny_fdtd // 2
mid_z_fdtd = nz_fdtd // 2
focal_idx_fdtd = np.argmax(pI_fdtd[:, mid_y_fdtd, mid_z_fdtd])
focal_z_fdtd = fdtd_zaxis[focal_idx_fdtd]
print(f'FDTD focal: z={focal_z_fdtd*1e3:.1f} mm')

# Interpolate FDTD onto ASM grids
interp_fdtd_xz = RegularGridInterpolator(
    (fdtd_zaxis, fdtd_yaxis), pI_fdtd[:, :, mid_z_fdtd],
    method='linear', bounds_error=False, fill_value=0)
interp_fdtd_yz = RegularGridInterpolator(
    (fdtd_zaxis, fdtd_xaxis), pI_fdtd[:, mid_y_fdtd, :],
    method='linear', bounds_error=False, fill_value=0)

z_common = zaxis[zaxis <= fdtd_zaxis[-1]]
nZ_c = len(z_common)

# FDTD x-z plane (at mid-elevation) on ASM grid
z_pts, x_pts = np.meshgrid(z_common, xaxis, indexing='ij')
fdtd_xz = interp_fdtd_xz(np.stack([z_pts.ravel(), x_pts.ravel()], axis=-1)).reshape(nZ_c, nX)

# FDTD y-z plane (at mid-lateral) on ASM grid
z_pts2, y_pts = np.meshgrid(z_common, yaxis, indexing='ij')
fdtd_yz = interp_fdtd_yz(np.stack([z_pts2.ravel(), y_pts.ravel()], axis=-1)).reshape(nZ_c, nY)

# FDTD focal plane on ASM grid
interp_fdtd_focal = RegularGridInterpolator(
    (fdtd_yaxis, fdtd_xaxis), pI_fdtd[focal_idx_fdtd, :, :],
    method='linear', bounds_error=False, fill_value=0)
asm_pts = np.meshgrid(xaxis, yaxis, indexing='ij')
fdtd_focal = interp_fdtd_focal(
    np.stack([asm_pts[0].ravel(), asm_pts[1].ravel()], axis=-1)).reshape(nX, nY)

# ASM x-z and y-z planes
asm_xz = pI[:, mid_y, :nZ_c].T   # (nZ_c, nX)
asm_yz = pI[mid_x, :, :nZ_c].T   # (nZ_c, nY)
asm_xz_h = pI_h[:, mid_y, :nZ_c].T
asm_yz_h = pI_h[mid_x, :, :nZ_c].T

ext_lat = [xaxis[0]*1e3, xaxis[-1]*1e3, yaxis[0]*1e3, yaxis[-1]*1e3]

# =========================================================================
# Figure 1: dB focal planes (FDTD, ASM skull, ASM homo)
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, data, title in zip(axes,
    [fdtd_focal, pI[:,:,focal_idx_s], pI_h[:,:,focal_idx_h]],
    [f'FDTD (z={focal_z_fdtd*1e3:.1f} mm)',
     f'ASM skull (z={zaxis[focal_idx_s]*1e3:.1f} mm)',
     f'ASM homo (z={zaxis[focal_idx_h]*1e3:.1f} mm)']):
    im = ax.imshow(to_dB(data).T, extent=ext_lat, aspect='equal',
                   vmin=-40, vmax=0, cmap='hot')
    ax.set(xlabel='x (mm)', ylabel='y (mm)', title=title)
    plt.colorbar(im, ax=ax, label='dB')

fig.suptitle('Focal Plane Intensity (dB)', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'dB_focal_planes.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print('  saved dB_focal_planes.png')

# =========================================================================
# Figure 2: dB x-z beamplots — depth on y-axis, zero at top
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ext_xz = [xaxis[0]*1e3, xaxis[-1]*1e3, z_common[-1]*1e3, 0]  # y: depth, origin=top

for ax, data, title in zip(axes,
    [fdtd_xz, asm_xz, asm_xz_h],
    ['FDTD x-z', 'ASM skull x-z', 'ASM homo x-z']):
    dB = to_dB(data)
    im = ax.imshow(dB, aspect='auto', extent=ext_xz,
                   vmin=-40, vmax=0, cmap='hot', origin='upper')
    ax.set(xlabel='Lateral (mm)', ylabel='Depth (mm)', title=title)
    plt.colorbar(im, ax=ax, label='dB')

fig.suptitle('Axial Beam Profile x-z (dB)', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'dB_xz_beams.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print('  saved dB_xz_beams.png')

# =========================================================================
# Figure 2b: dB y-z beamplots — depth on y-axis, zero at top
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ext_yz = [yaxis[0]*1e3, yaxis[-1]*1e3, z_common[-1]*1e3, 0]

for ax, data, title in zip(axes,
    [fdtd_yz, asm_yz, asm_yz_h],
    ['FDTD y-z', 'ASM skull y-z', 'ASM homo y-z']):
    dB = to_dB(data)
    im = ax.imshow(dB, aspect='auto', extent=ext_yz,
                   vmin=-40, vmax=0, cmap='hot', origin='upper')
    ax.set(xlabel='Elevation (mm)', ylabel='Depth (mm)', title=title)
    plt.colorbar(im, ax=ax, label='dB')

fig.suptitle('Axial Beam Profile y-z (dB)', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'dB_yz_beams.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print('  saved dB_yz_beams.png')

# =========================================================================
# Figure 3: Beam profiles (dB) — on-axis, lateral, elevational, -6dB contour
# =========================================================================
# On-axis profiles
onaxis_fdtd_raw = pI_fdtd[:, mid_y_fdtd, mid_z_fdtd]
interp_onaxis = RegularGridInterpolator(
    (fdtd_zaxis,), onaxis_fdtd_raw, method='linear',
    bounds_error=False, fill_value=0)
fdtd_onaxis = interp_onaxis(z_common[:, None]).ravel()
asm_onaxis_s = pI[mid_x, mid_y, :nZ_c]
asm_onaxis_h = pI_h[mid_x, mid_y, :nZ_c]

# Normalize all to FDTD peak for absolute comparison
peak_fdtd = np.max(fdtd_onaxis) + 1e-30

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# On-axis (dB)
axes[0, 0].plot(z_common*1e3, 10*np.log10(fdtd_onaxis/peak_fdtd + 1e-30), 'b-', lw=2, label='FDTD')
axes[0, 0].plot(z_common*1e3, 10*np.log10(asm_onaxis_s/peak_fdtd + 1e-30), 'r--', lw=2, label='ASM (skull)')
axes[0, 0].plot(z_common*1e3, 10*np.log10(asm_onaxis_h/peak_fdtd + 1e-30), 'g:', lw=2, label='ASM (homo)')
axes[0, 0].set(xlabel='Depth (mm)', ylabel='dB', title='On-axis intensity', ylim=(-40, 5))
axes[0, 0].legend(); axes[0, 0].grid(True)

# Lateral (x) at focus — each normalized to its own peak
fdtd_lat = fdtd_focal[:, nY//2]
asm_lat_s = pI[:, mid_y, focal_idx_s]
asm_lat_h = pI_h[:, mid_y, focal_idx_h]
axes[0, 1].plot(xaxis*1e3, to_dB(fdtd_lat), 'b-', lw=2, label='FDTD')
axes[0, 1].plot(xaxis*1e3, to_dB(asm_lat_s), 'r--', lw=2, label='ASM (skull)')
axes[0, 1].plot(xaxis*1e3, to_dB(asm_lat_h), 'g:', lw=2, label='ASM (homo)')
axes[0, 1].set(xlabel='Lateral x (mm)', ylabel='dB', title='Lateral beam (x) at focus', ylim=(-40, 2))
axes[0, 1].legend(); axes[0, 1].grid(True)

# Elevational (y) at focus
fdtd_elev = fdtd_focal[nX//2, :]
asm_elev_s = pI[mid_x, :, focal_idx_s]
asm_elev_h = pI_h[mid_x, :, focal_idx_h]
axes[1, 0].plot(yaxis*1e3, to_dB(fdtd_elev), 'b-', lw=2, label='FDTD')
axes[1, 0].plot(yaxis*1e3, to_dB(asm_elev_s), 'r--', lw=2, label='ASM (skull)')
axes[1, 0].plot(yaxis*1e3, to_dB(asm_elev_h), 'g:', lw=2, label='ASM (homo)')
axes[1, 0].set(xlabel='Elevation y (mm)', ylabel='dB', title='Lateral beam (y) at focus', ylim=(-40, 2))
axes[1, 0].legend(); axes[1, 0].grid(True)

# -6 dB contours at focal plane
ax = axes[1, 1]
fdtd_foc_dB = to_dB(fdtd_focal)
asm_foc_dB_s = to_dB(pI[:,:,focal_idx_s])
asm_foc_dB_h = to_dB(pI_h[:,:,focal_idx_h])
ax.contour(xaxis*1e3, yaxis*1e3, fdtd_foc_dB.T, levels=[-6], colors='b', linewidths=2)
ax.contour(xaxis*1e3, yaxis*1e3, asm_foc_dB_s.T, levels=[-6], colors='r', linestyles='--', linewidths=2)
ax.contour(xaxis*1e3, yaxis*1e3, asm_foc_dB_h.T, levels=[-6], colors='g', linestyles=':', linewidths=2)
ax.set(xlabel='x (mm)', ylabel='y (mm)', title='-6 dB focal contours', aspect='equal')
# Manual legend for contours
from matplotlib.lines import Line2D
ax.legend([Line2D([0],[0],color='b',lw=2), Line2D([0],[0],color='r',ls='--',lw=2),
           Line2D([0],[0],color='g',ls=':',lw=2)], ['FDTD','ASM (skull)','ASM (homo)'])

fig.suptitle('Beam Profiles (dB)', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'dB_beam_profiles.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print('  saved dB_beam_profiles.png')

# =========================================================================
# Figure 4: FDTD vs ASM comparison (updated with depth on y-axis)
# =========================================================================
fdtd_norm = fdtd_focal / (np.max(fdtd_focal) + 1e-30)
asm_norm = pI[:,:,focal_idx_s] / (np.max(pI[:,:,focal_idx_s]) + 1e-30)
diff = np.abs(fdtd_norm - asm_norm)
rms_diff = np.sqrt(np.mean(diff**2))

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Row 1: focal planes
im = axes[0,0].imshow(fdtd_norm.T, extent=ext_lat, aspect='equal', vmin=0, vmax=1)
axes[0,0].set(xlabel='x (mm)', ylabel='y (mm)',
              title=f'FDTD focal (z={focal_z_fdtd*1e3:.1f} mm)')
plt.colorbar(im, ax=axes[0,0], label='normalized')

im = axes[0,1].imshow(asm_norm.T, extent=ext_lat, aspect='equal', vmin=0, vmax=1)
axes[0,1].set(xlabel='x (mm)', ylabel='y (mm)',
              title=f'ASM focal (z={zaxis[focal_idx_s]*1e3:.1f} mm)')
plt.colorbar(im, ax=axes[0,1], label='normalized')

im = axes[0,2].imshow(diff.T, extent=ext_lat, aspect='equal', vmin=0, cmap='hot')
axes[0,2].set(xlabel='x (mm)', ylabel='y (mm)',
              title=f'|FDTD - ASM| (RMS={rms_diff:.3f})')
plt.colorbar(im, ax=axes[0,2])

# Row 2: on-axis, lateral, x-z difference — depth on y-axis
fdtd_onaxis_norm = fdtd_onaxis / (np.max(fdtd_onaxis) + 1e-30)
asm_onaxis_norm = asm_onaxis_s / (np.max(asm_onaxis_s) + 1e-30)

axes[1,0].plot(z_common*1e3, fdtd_onaxis_norm, 'b-', lw=2, label='FDTD')
axes[1,0].plot(z_common*1e3, asm_onaxis_norm, 'r--', lw=2, label='ASM')
axes[1,0].set(xlabel='Depth (mm)', ylabel='Normalized intensity', title='On-axis intensity')
axes[1,0].legend(); axes[1,0].grid(True)

fdtd_lat_n = fdtd_focal[:, nY//2] / (np.max(fdtd_focal[:, nY//2]) + 1e-30)
asm_lat_n = pI[:, mid_y, focal_idx_s] / (np.max(pI[:, mid_y, focal_idx_s]) + 1e-30)
axes[1,1].plot(xaxis*1e3, fdtd_lat_n, 'b-', lw=2, label='FDTD')
axes[1,1].plot(xaxis*1e3, asm_lat_n, 'r--', lw=2, label='ASM')
axes[1,1].set(xlabel='x (mm)', ylabel='Normalized intensity', title='Lateral profile at focus')
axes[1,1].legend(); axes[1,1].grid(True)

# x-z difference with depth on y-axis, zero at top
fdtd_xz_norm = fdtd_xz / (np.max(fdtd_xz) + 1e-30)
asm_xz_norm = asm_xz / (np.max(asm_xz) + 1e-30)
xz_diff = np.abs(fdtd_xz_norm - asm_xz_norm)
im = axes[1,2].imshow(xz_diff, aspect='auto', extent=ext_xz,
                       vmin=0, cmap='hot', origin='upper')
axes[1,2].set(xlabel='Lateral (mm)', ylabel='Depth (mm)',
              title=f'|FDTD - ASM| x-z (RMS={np.sqrt(np.mean(xz_diff**2)):.3f})')
plt.colorbar(im, ax=axes[1,2])

fig.suptitle('FDTD vs ASM Transcranial Comparison (normalized)', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fdtd_vs_asm.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print('  saved fdtd_vs_asm.png')

# =========================================================================
# Figure 5: ASM-only transcranial comparison (homo vs skull) — depth on y-axis
# =========================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

im = axes[0,0].imshow(pI_h[:,:,focal_idx_h].T, extent=ext_lat, aspect='equal')
axes[0,0].set(xlabel='x (mm)', ylabel='y (mm)',
              title=f'Homogeneous focal (z={zaxis[focal_idx_h]*1e3:.1f} mm)')
plt.colorbar(im, ax=axes[0,0])

im = axes[0,1].imshow(pI[:,:,focal_idx_s].T, extent=ext_lat, aspect='equal')
axes[0,1].set(xlabel='x (mm)', ylabel='y (mm)',
              title=f'Through skull focal (z={zaxis[focal_idx_s]*1e3:.1f} mm)')
plt.colorbar(im, ax=axes[0,1])

ratio = pI[:,:,focal_idx_s] / (pI_h[:,:,focal_idx_h] + 1e-10)
im = axes[0,2].imshow(ratio.T, extent=ext_lat, aspect='equal', vmin=0, vmax=1, cmap='hot')
axes[0,2].set(xlabel='x (mm)', ylabel='y (mm)', title='Skull / Homogeneous ratio')
plt.colorbar(im, ax=axes[0,2])

axes[1,0].plot(zaxis*1e3, pI_h[mid_x, mid_y, :], label='Homogeneous')
axes[1,0].plot(zaxis*1e3, pI[mid_x, mid_y, :], label='Through skull')
axes[1,0].set(xlabel='Depth (mm)', ylabel='On-axis intensity', title='On-axis intensity vs depth')
axes[1,0].legend(); axes[1,0].grid(True)

focal_loss_dB = d['focal_loss_dB']

# x-z beamplots with depth on y-axis, zero at top
ext_xz_full = [xaxis[0]*1e3, xaxis[-1]*1e3, zaxis[-1]*1e3, 0]
im = axes[1,1].imshow(pI_h[:, mid_y, :].T, aspect='auto', extent=ext_xz_full, origin='upper')
axes[1,1].set(xlabel='Lateral (mm)', ylabel='Depth (mm)', title='Homogeneous x-z')
plt.colorbar(im, ax=axes[1,1])

im = axes[1,2].imshow(pI[:, mid_y, :].T, aspect='auto', extent=ext_xz_full, origin='upper')
axes[1,2].set(xlabel='Lateral (mm)', ylabel='Depth (mm)', title='Through skull x-z')
plt.colorbar(im, ax=axes[1,2])

fig.suptitle(f'Transcranial ASM: f0=1 MHz, focal=50 mm, p0=1.5 MPa '
             f'(skull loss: {float(focal_loss_dB):.1f} dB)', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'transcranial_comparison_asm_only.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print('  saved transcranial_comparison_asm_only.png')

print('\nAll figures regenerated.')
