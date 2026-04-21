"""TOF methods on a larger-aperture aberrated planewave.

Builds a planewave (flat base delay, no focusing) on a circular
Hanning-apodized aperture of radius 8 mm, adds a random aberration
screen with 50 ns RMS and 10-lambda coherence length, propagates
through the angular spectrum solver, and compares the same 5 TOF
methods as the simplest-case figure.

Ground truth for each depth is the geometric min-over-aperture of
``delay_src(x_s, y_s) + |r - r_src| / c0`` — same as the focused test.
Because the source now has aberration, the ground-truth surface is no
longer flat; it encodes the aberration + geometric spread.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(_here)
sys.path.insert(0, _repo)
sys.path.insert(0, _here)

from angular_spectrum_solver import angular_spectrum_solve, SolverParams  # noqa: E402
from tof_extraction import (  # noqa: E402
    extract_tof_envelope, extract_tof_matched_filter_parabolic)


def build_aberrated_planewave(nX, nY, nT, dX, dY, dT, f0, c0, p0,
                              R_m, ncycles, dur, t_lead_s,
                              aberr_rms_s, coherence_lambdas, rng_seed):
    """Circular aperture, Hanning apodization, planewave (flat delay)
    plus a smooth random aberration screen.

    The aberration screen is white Gaussian noise low-pass filtered
    with a Gaussian of FWHM = ``coherence_lambdas * lambda``, then
    rescaled to ``aberr_rms_s`` RMS over the active aperture.
    """
    lam = c0 / f0
    xaxis = (np.arange(nX) - nX / 2 + 0.5) * dX
    yaxis = (np.arange(nY) - nY / 2 + 0.5) * dY
    XX, YY = np.meshgrid(xaxis, yaxis, indexing='ij')
    r2 = XX ** 2 + YY ** 2
    r = np.sqrt(r2)
    aperture = (r2 <= R_m ** 2).astype(np.float32)
    # Edge-only Hanning apodization: apod = 1 in the interior, tapers
    # down to 0.5 at the rim using the middle slice of a Hanning wider
    # than the edge band (hanning_full_m = 2 * edge_width_m → rim = 0.5).
    edge_width_pix = 15
    edge_width_m = edge_width_pix * dX
    hanning_full_m = 2.0 * edge_width_m
    offset = R_m - r                            # 0 at the rim, grows inward
    in_edge_band = (r <= R_m) & (offset < edge_width_m)
    taper = np.where(
        in_edge_band,
        0.5 * (1 + np.cos(np.pi * (edge_width_m - offset) / hanning_full_m)),
        1.0,
    )
    apod = np.where(r <= R_m, taper, 0.0).astype(np.float32)

    # aberration screen
    rng = np.random.default_rng(rng_seed)
    noise = rng.standard_normal((nX, nY))
    # FWHM = coherence_lambdas * lambda, sigma_pix = FWHM / (2 sqrt(2 ln 2) * dX)
    fwhm_m = coherence_lambdas * lam
    sigma_pix_x = fwhm_m / (2.0 * np.sqrt(2.0 * np.log(2.0)) * dX)
    sigma_pix_y = fwhm_m / (2.0 * np.sqrt(2.0 * np.log(2.0)) * dY)
    aberr = gaussian_filter(noise, (sigma_pix_x, sigma_pix_y))
    # normalize to desired RMS over the active aperture
    mask = aperture > 0
    aberr -= aberr[mask].mean()
    rms_now = np.sqrt(np.mean(aberr[mask] ** 2))
    aberr *= (aberr_rms_s / max(rms_now, 1e-30))

    # base planewave delay = constant t_lead; add aberration only inside aperture
    delays = np.full_like(r, t_lead_s, dtype=np.float64)
    delays[mask] = t_lead_s + aberr[mask]

    omega0 = 2 * np.pi * f0
    t_axis = np.arange(nT) * dT
    t_grid = t_axis[None, None, :]
    d_grid = delays[:, :, None]
    envelope = np.exp(-(1.05 * (t_grid - d_grid) * omega0
                        / (ncycles * np.pi)) ** (2 * dur))
    field = (apod[:, :, None] * envelope
             * np.sin(omega0 * (t_grid - d_grid)) * p0).astype(np.float32)
    return (field, aperture.astype(np.float32), t_axis,
            delays.astype(np.float64), aberr.astype(np.float64),
            apod.astype(np.float32))


def geometric_tof_groundtruth(aperture, delays, dX, dY, c0, z, nX, nY):
    ax = (np.arange(nX) - nX / 2 + 0.5) * dX
    ay = (np.arange(nY) - nY / 2 + 0.5) * dY
    active = aperture > 0
    AX, AY = np.meshgrid(ax, ay, indexing='ij')
    sx = AX[active]
    sy = AY[active]
    sd = delays[active]
    tof = np.full((nX, nY), np.inf, dtype=np.float64)
    rx = ax[:, None].repeat(nY, axis=1)
    ry = ay[None, :].repeat(nX, axis=0)
    for k in range(sx.size):
        d = np.sqrt((rx - sx[k]) ** 2 + (ry - sy[k]) ** 2 + z ** 2)
        t = sd[k] + d / c0
        tof = np.minimum(tof, t)
    return tof.astype(np.float32)


def main():
    # --- grid + aberrated planewave -----------------------------------
    f0 = 1e6
    c0 = 1500.0
    lam = c0 / f0                  # 1.5 mm
    dX = dY = lam / 5              # 0.30 mm
    # Aperture R = 8 mm (was 2.5 mm); need nX s.t. FOV covers aperture + buffer
    R_m = 5e-3
    # Lateral domain from -15 mm to +15 mm (30 mm span, 100 pixels at dX = 0.3 mm)
    nX = int(round(30e-3 / dX))
    nX += nX % 2
    nY = nX
    # 8 mm max propagation (~5.3 us of lab-frame travel) fits comfortably
    # in a 512-sample, 20.48 us window with the pulse centred at 10.24 us.
    nT = 512
    dT = 4e-8                      # 25 MHz
    p0 = 1e3
    boundary_factor = 0.2
    t_bw = max(round(nT * boundary_factor), 1)
    # aberrated planewave: peak of aberration can push a trace ~4-5 RMS late
    aberr_rms_s = 100e-9
    coherence_lambdas = 2.0
    # Centre the initial pulse in the time window. With 8 mm max
    # propagation (~5.3 us lab-frame delay) the pulse at z=8 mm sits
    # at ~15.5 us, well inside the ~16.4 us right-hand non-ABL edge.
    t_lead_min = t_bw * dT + 0.5 / f0 + 3 * aberr_rms_s
    t_lead = 0.5 * nT * dT
    assert t_lead >= t_lead_min, (
        f't_lead {t_lead*1e6:.2f} us < minimum {t_lead_min*1e6:.2f} us')
    print(f't_lead = {t_lead*1e6:.2f} us  (min safe = {t_lead_min*1e6:.2f} us; '
          f'nT*dT = {nT*dT*1e6:.2f} us)')

    print(f'grid = {nX}x{nY}x{nT};  FOV = {nX*dX*1e3:.2f} mm; '
          f'dX = {dX*1e3:.3f} mm (lam/5);  aperture R = {R_m*1e3:.1f} mm')
    print(f'aberration: rms = {aberr_rms_s*1e9:.1f} ns,  '
          f'coherence FWHM = {coherence_lambdas:.1f} lambda = '
          f'{coherence_lambdas*lam*1e3:.2f} mm')

    field0, aperture, taxis, delays, aberr_map, apod = build_aberrated_planewave(
        nX, nY, nT, dX, dY, dT, f0, c0, p0, R_m,
        ncycles=1, dur=2, t_lead_s=t_lead,
        aberr_rms_s=aberr_rms_s,
        coherence_lambdas=coherence_lambdas,
        rng_seed=20260421)
    # verify aberration RMS
    mask = aperture > 0
    print(f'achieved aberration RMS over aperture = '
          f'{np.sqrt(np.mean(aberr_map[mask] ** 2))*1e9:.2f} ns')

    params = SolverParams(
        dX=dX, dY=dY, dT=dT, c0=c0, rho0=1000., beta=3.5,
        alpha0=-1., f0=f0, propDist=8e-3,
        boundaryFactor=boundary_factor,
        useSplitStep=True, useAdaptiveFiltering=False, useTVD=False,
        fluxScheme='rusanov', boundaryProfile='quadratic',
        useFreqWeightedBoundary=False, useSuperAbsorbing=False,
        dZmin=1e-3, stabilityThreshold=1.0)

    step_fields = []
    def cb(cc, z, fld):
        step_fields.append(np.asarray(fld).copy())
    # Only zaxis is consumed downstream; per-step fields are captured by
    # the callback into ``step_fields``, from which the TOF-method
    # comparison runs independently.
    zaxis = angular_spectrum_solve(
        field0, params, verbose=False, per_step_callback=cb)[5]
    print(f'nZ = {len(zaxis)}; zaxis (mm) = {zaxis*1e3}')

    # Reference waveform for xcorr: use the NON-aberrated planewave pulse
    # (i.e., the pulse shape without delay variation), taken at the centre.
    # We take the source trace at an aperture pixel near the centre; aberration
    # there is small but non-zero — good enough for template xcorr since xcorr
    # is a lag estimator.
    cx, cy = nX // 2, nY // 2
    ref_trace = field0[cx, cy, :].astype(np.float32)
    from scipy.signal import hilbert as _hilb
    t_ref_peak = float(taxis[np.argmax(np.abs(_hilb(ref_trace)))])
    print(f'ref pulse peaks at t = {t_ref_peak*1e6:.3f} us')

    methods = [
        ('baseline: envelope argmax',
         lambda f, z: extract_tof_envelope(f, taxis, z, c0, ratio=1.0)),
        ('M1: 10%-of-peak leading edge',
         lambda f, z: extract_tof_envelope(f, taxis, z, c0, ratio=0.1)),
        ('M2: xcorr + parabolic refine',
         lambda f, z: extract_tof_matched_filter_parabolic(
             f, taxis, z, c0, ref_trace, t_ref_peak)),
    ]

    depth_idx = np.linspace(0, len(zaxis) - 1, 4).round().astype(int)
    nrows = len(methods)
    ncols = len(depth_idx)

    print('computing ground truth per depth ...')
    tof_maps = [[method(step_fields[zi], float(zaxis[zi]))
                 for zi in range(len(zaxis))]
                for _name, method in methods]
    gt = [geometric_tof_groundtruth(aperture, delays, dX, dY, c0,
                                    float(zaxis[zi]), nX, nY)
          for zi in range(len(zaxis))]

    amp_per_z = [np.abs(step_fields[zi]).max(axis=-1)
                 for zi in range(len(zaxis))]

    def rms_err(tof, truth, amp, amp_thresh):
        m = (amp >= amp_thresh * amp.max()) & np.isfinite(tof) & np.isfinite(truth)
        if not m.any():
            return np.nan
        diff = (tof - truth)[m]
        return float(np.sqrt(np.mean(diff ** 2)))

    # --- M1 (10% leading edge) has a known systematic bias: its TOF
    # fires ~half-pulse-width earlier than the envelope peak. Remove
    # that bias with a single GLOBAL offset (one scalar across all
    # z-slices), computed as the median of (M1 - baseline) over all
    # gate-10% voxels pooled across depth. This exposes any residual
    # depth-dependent structure in M1's error.
    m1_idx = 1
    baseline_idx = 0
    tof_maps_biascorr = [[tm.copy() for tm in row] for row in tof_maps]
    diffs = []
    for zi in range(len(zaxis)):
        amp = amp_per_z[zi]
        m = amp >= 0.1 * amp.max()
        d = tof_maps[m1_idx][zi] - tof_maps[baseline_idx][zi]
        diffs.append(d[m])
    pooled = np.concatenate([d.ravel() for d in diffs])
    pooled = pooled[np.isfinite(pooled)]
    bias_global = float(np.nanmedian(pooled)) if pooled.size else 0.0
    for zi in range(len(zaxis)):
        tof_maps_biascorr[m1_idx][zi] = tof_maps[m1_idx][zi] - bias_global
    # keep a per-slice list for the plot annotations (all equal to the global)
    bias_per_z = [bias_global] * len(zaxis)
    print(f'M1 global bias (ns, median of pooled M1 - baseline over gate-10% voxels):  '
          f'{bias_global*1e9:+.1f}')

    gate_thresholds = [0.0, 0.01, 0.1, 0.3, 0.5]
    err_by_gate = {}
    for g in gate_thresholds:
        table = []
        for ri in range(nrows):
            src = tof_maps_biascorr if ri == m1_idx else tof_maps
            errs = [rms_err(src[ri][zi], gt[zi], amp_per_z[zi], g) * 1e9
                    for zi in range(len(zaxis))]
            table.append(errs)
        err_by_gate[g] = table
    err_by_method = err_by_gate[0.1]

    # --- figure ---------------------------------------------------------
    fig = plt.figure(figsize=(4.5 * ncols, 2.6 * nrows + 3.0 + 2.6),
                     facecolor='black')
    gs = fig.add_gridspec(nrows + 2, ncols,
                          height_ratios=[1.3] + [1] * nrows + [1.15],
                          hspace=0.5, wspace=0.28)

    xaxis_mm = (np.arange(nX) - nX / 2 + 0.5) * dX * 1e3
    t_us = taxis * 1e6
    t_abl_low = float(taxis[t_bw - 1]) * 1e6
    t_abl_high = float(taxis[nT - t_bw]) * 1e6

    # --- top row: 4 source-definition maps (aperture, apod, aberration, total delay) ---
    ab_vmax = float(np.nanmax(np.abs(aberr_map))) * 1e9
    # total delay minus the constant lead, shown in ns, only inside aperture
    dshow = (delays - float(t_lead)) * 1e9
    dshow = np.where(aperture > 0, dshow, np.nan)
    d_vmax = float(np.nanmax(np.abs(dshow)))

    top_panels = [
        ('Aperture mask', aperture, 'gray', None, None, ''),
        ('Apodization (edge-only Hanning, 15 pix)',
         apod, 'viridis', 0.0, 1.0, ''),
        (f'Aberration (ns)  RMS={aberr_rms_s*1e9:.0f},  coh={coherence_lambdas:.0f}λ',
         aberr_map * 1e9, 'RdBu_r', -ab_vmax, ab_vmax, 'ns'),
        ('Total delay − t_lead (ns)', dshow, 'RdBu_r', -d_vmax, d_vmax, 'ns'),
    ]
    for ci_top, (title, data, cmap, vmin, vmax, cblabel) in enumerate(top_panels):
        ax = fig.add_subplot(gs[0, ci_top])
        im = ax.imshow(data.T, origin='lower',
                       extent=(xaxis_mm[0], xaxis_mm[-1],
                               xaxis_mm[0], xaxis_mm[-1]),
                       cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        ax.contour(xaxis_mm, xaxis_mm, aperture.T, levels=[0.5],
                   colors='white', linewidths=0.6)
        ax.set_title(title, color='white', fontsize=11)
        ax.set_xlabel('x (mm)', color='white', fontsize=10)
        if ci_top == 0:
            ax.set_ylabel('y (mm)', color='white', fontsize=10)
        ax.tick_params(colors='white', labelsize=9)
        for s in ax.spines.values():
            s.set_color('white')
        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cb.ax, 'yticklabels'), color='white')

    # --- method rows (xt slice, auto-zoomed, amplitude-gated) ---------
    xt_max = max(np.abs(step_fields[zi][:, cy, :]).max() for zi in depth_idx)
    amp_gate = 0.1
    zoom_half_us = 1.5
    zoom_half_mm = R_m * 1e3 * 1.1

    for ri, (name, _method) in enumerate(methods):
        for ci, zi in enumerate(depth_idx):
            ax = fig.add_subplot(gs[1 + ri, ci])
            fld = step_fields[zi][:, cy, :]
            ax.imshow(fld.T, origin='lower', aspect='auto',
                      cmap='seismic', vmin=-xt_max, vmax=xt_max,
                      extent=(xaxis_mm[0], xaxis_mm[-1],
                              t_us[0], t_us[-1]))
            z_here = float(zaxis[zi])
            amp_slice = np.abs(step_fields[zi][:, cy, :]).max(axis=-1)
            amp_mask = amp_slice >= amp_gate * amp_slice.max()

            tof_line_full = (tof_maps[ri][zi][:, cy] - z_here / c0) * 1e6
            tof_line = np.where(amp_mask, tof_line_full, np.nan)
            ax.plot(xaxis_mm, tof_line, color='#ff3030', lw=1.6,
                    label='raw' if ri == m1_idx else None)

            # For M1, overlay the bias-corrected curve so it sits on the pulse peak.
            if ri == m1_idx:
                corr_full = (tof_maps_biascorr[ri][zi][:, cy]
                             - z_here / c0) * 1e6
                corr_line = np.where(amp_mask, corr_full, np.nan)
                ax.plot(xaxis_mm, corr_line, color='#33cc66', lw=1.6,
                        label=f'global bias-corr ({bias_global*1e9:+.0f} ns)')
                leg = ax.legend(facecolor='black', edgecolor='white',
                                labelcolor='white', fontsize=8, loc='upper right')
                leg.get_frame().set_alpha(0.75)

            gt_line_full = (gt[zi][:, cy] - z_here / c0) * 1e6
            gt_line = np.where(amp_mask, gt_line_full, np.nan)
            ax.plot(xaxis_mm, gt_line, color='white', lw=1.0,
                    ls='--', alpha=0.85)

            ax.axhspan(t_us[0], t_abl_low, color='#ffa500',
                       alpha=0.12, zorder=0.5)
            ax.axhspan(t_abl_high, t_us[-1], color='#ffa500',
                       alpha=0.12, zorder=0.5)

            t_center = float(np.nanmedian(gt_line))
            if np.isfinite(t_center):
                ax.set_ylim(t_center - zoom_half_us,
                            t_center + zoom_half_us)
            ax.set_xlim(-zoom_half_mm, zoom_half_mm)

            if ri == 0:
                ax.set_title(f'z = {z_here*1e3:.1f} mm',
                             color='white', fontsize=12)
            if ci == 0:
                ax.set_ylabel(f'{name}\nt (us)',
                              color='white', fontsize=10)
            if ri == len(methods) - 1:
                ax.set_xlabel('x (mm)', color='white', fontsize=10)
            ax.tick_params(colors='white', labelsize=9)
            for s in ax.spines.values():
                s.set_color('white')
            ax.set_facecolor('black')

    # --- bottom row: RMS error vs depth ---------------------------------
    ax_err = fig.add_subplot(gs[-1, :])
    colors = ['#cccccc', '#ffb454', '#4fc3f7', '#66bb6a', '#ef5350']
    zmm = zaxis * 1e3
    for ri, (name, _m) in enumerate(methods):
        lbl = name + (f' (global bias {bias_global*1e9:+.0f} ns)' if ri == m1_idx else '')
        ax_err.plot(zmm, err_by_method[ri], '-o',
                    color=colors[ri], lw=1.6, ms=4, label=lbl)
    ax_err.set_xlabel('z (mm)', color='white', fontsize=11)
    ax_err.set_ylabel('RMS TOF error (ns)\n(|amp| >= 10% per slice)',
                      color='white', fontsize=10)
    ax_err.set_title(
        'TOF error vs geometric first-arrival ground truth  '
        f'(aberrated planewave: RMS={aberr_rms_s*1e9:.0f} ns, '
        f'coh={coherence_lambdas:.0f}λ)',
        color='white', fontsize=12)
    ax_err.set_yscale('log')
    ax_err.grid(True, which='both', color='#444', lw=0.5)
    ax_err.tick_params(colors='white', labelsize=10)
    ax_err.set_facecolor('black')
    for s in ax_err.spines.values():
        s.set_color('white')
    leg = ax_err.legend(facecolor='black', edgecolor='white',
                        labelcolor='white', fontsize=10, loc='best')
    leg.get_frame().set_alpha(0.8)

    fig.suptitle(
        'TOF methods on an aberrated planewave  '
        f'(R={R_m*1e3:.1f} mm, 1-cycle, water).  '
        'Red = method TOF-z/c0; white dashed = geometric ground truth.',
        color='white', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = os.path.join(_here, 'figures',
                            'tof_methods_aberrated_planewave_100ns_r5.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, facecolor='black', dpi=110,
                bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')

    # print RMS error summary
    header = ['method'] + [f'gate {int(g*100):>2d}%' for g in gate_thresholds]
    print('\nRMS TOF error (ns), mean across all depths, by amplitude gate:')
    print('  ' + '  '.join(f'{h:>24s}' if i == 0 else f'{h:>9s}'
                            for i, h in enumerate(header)))
    for ri, (name, _m) in enumerate(methods):
        row_vals = [np.nanmean(np.array(err_by_gate[g][ri])) for g in gate_thresholds]
        cells = [f'{name:>24s}'] + [f'{v:9.2f}' for v in row_vals]
        print('  ' + '  '.join(cells))

    print('\nRMS TOF error (ns), median across all depths, by amplitude gate:')
    print('  ' + '  '.join(f'{h:>24s}' if i == 0 else f'{h:>9s}'
                            for i, h in enumerate(header)))
    for ri, (name, _m) in enumerate(methods):
        row_vals = [np.nanmedian(np.array(err_by_gate[g][ri])) for g in gate_thresholds]
        cells = [f'{name:>24s}'] + [f'{v:9.2f}' for v in row_vals]
        print('  ' + '  '.join(cells))


if __name__ == '__main__':
    main()
