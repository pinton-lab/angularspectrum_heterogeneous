"""
Test 11d — Nonlinear shocked-focus k-space filter showcase.

Extends the focused-piston geometry of Test 11b (8-lambda aperture,
F = 20 mm) by enabling finite-amplitude propagation (beta > 0) with a
drive level high enough that the pulse steepens toward a shock near the
geometric focus.  Runs the solver twice — with and without the adaptive
k-space filter — and compares:

  (1) on-axis waveform at focus (time domain): the unfiltered run
      typically shows high-frequency ringing behind the shock as
      harmonic energy wraps into the transverse spectrum and aliases;
  (2) on-axis magnitude spectrum at focus: harmonic decay vs. noise
      floor / fold-back;
  (3) peak pressure vs z: confirms the filter does not clip the
      physically-relevant part of the spectrum.

Results saved to validation_results/kfilter/.
"""

import os, sys, time
import numpy as np
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


def run_nonlinear_shocked_focus(nT=1401, ncycles=6.0, p0=1.5e6, beta=3.5,
                                 dZ_factor=2, tag=''):
    """Nonlinear focused piston, filter on vs off.

    Parameters
    ----------
    nT : int
        Number of time samples.  Must be long enough that the pulse plus
        focal delay spread fits inside the window without wrap-around.
    ncycles : float
        Pulse length (Gaussian-envelope cycles).  Longer pulses give
        more distance over which to steepen.
    p0 : float
        Source pressure amplitude (Pa).  Higher → stronger nonlinear
        harmonic generation.  1.5 MPa on an 8-lambda bowl at f0 = 4 MHz
        with beta = 3.5 develops visible shock content near focus.
    beta : float
        Nonlinearity coefficient (3.5 ≈ soft tissue).
    dZ_factor : int
        Axial step size in units of dX (forwarded to dZmin).
    tag : str
        Filename suffix.
    """
    print(f'\n=== Test 11d: Nonlinear shocked focus — k-filter on vs off '
          f'(p0={p0/1e6:.2f} MPa, ncycles={ncycles}, beta={beta}) ===')

    f0 = 4e6
    c0 = 1500.0
    rho0 = 1000.0
    lam = c0 / f0
    dX = lam / 5
    dT = dX / (5 * c0)
    nX, nY = 193, 193
    a = 8 * lam
    F_foc = 20e-3
    propDist = 1.1 * F_foc

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

    configs = [
        ('k-filter OFF', dict(useAdaptiveFiltering=False)),
        ('k-filter ON',  dict(useAdaptiveFiltering=True)),
    ]

    results = {}
    for label, overrides in configs:
        print(f'\n--- {label} ---')
        params = SolverParams(
            dX=dX, dY=dX, dT=dT, c0=c0, rho0=rho0,
            beta=beta, alpha0=-1, f0=f0, propDist=propDist,
            useSplitStep=True, useTVD=True,
            boundaryFactor=0.15, useBoundaryLayer=True,
            stabilityThreshold=0.2, stabilityRecoveryFactor=0.15,
            dZmin=dX * dZ_factor,
            **overrides)
        t0 = time.time()
        field_out, pnp, ppp, pI, pIloss, zaxis, pax = \
            angular_spectrum_solve(field0, params, verbose=False)
        dt_run = time.time() - t0
        zaxis = np.asarray(zaxis)
        pax_np = np.asarray(pax)

        idx_F = int(np.argmin(np.abs(zaxis - F_foc)))
        waveform_F = pax_np[:, idx_F]
        pmax_onaxis = np.max(np.abs(pax_np), axis=0)
        ppp_onaxis = np.max(pax_np, axis=0)
        pnp_onaxis = -np.min(pax_np, axis=0)

        # Magnitude spectrum at focus
        win = np.hanning(nT).astype(np.float32)
        W = np.fft.rfft(waveform_F * win)
        freqs = np.fft.rfftfreq(nT, d=dT)
        mag = np.abs(W) / np.max(np.abs(W) + 1e-30)

        print(f'    runtime        = {dt_run:.1f} s')
        print(f'    PPP at focus   = {ppp_onaxis[idx_F]/1e6:.3f} MPa')
        print(f'    PNP at focus   = {pnp_onaxis[idx_F]/1e6:.3f} MPa')
        print(f'    peak |p| max   = {pmax_onaxis.max()/1e6:.3f} MPa')

        results[label] = dict(
            zaxis=zaxis, pmax_onaxis=pmax_onaxis,
            ppp_onaxis=ppp_onaxis, pnp_onaxis=pnp_onaxis,
            waveform_F=waveform_F, taxis=taxis,
            freqs=freqs, spectrum=mag,
            idx_F=idx_F, runtime_s=dt_run,
        )

    # --- Figure: waveform at focus, spectrum, peak pressure vs z ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = {'k-filter OFF': 'C3', 'k-filter ON': 'C0'}

    # Waveform at focus (trimmed around the pulse)
    ref = results['k-filter ON']
    env = np.abs(ref['waveform_F'])
    peak_idx = int(np.argmax(env))
    half_window = int(1.5 * ncycles / (f0 * dT))
    i0 = max(peak_idx - half_window, 0)
    i1 = min(peak_idx + half_window, nT)

    for label, r in results.items():
        axes[0].plot(r['taxis'][i0:i1]*1e6, r['waveform_F'][i0:i1]/1e6,
                     color=colors[label], lw=1.1, label=label)
    axes[0].set(xlabel='t (µs)', ylabel='Pressure at focus (MPa)',
                title='On-axis waveform at z = F')
    axes[0].legend(); axes[0].grid(True)

    # Magnitude spectrum at focus (log, normalized to fundamental bin)
    for label, r in results.items():
        # Normalize each spectrum to its fundamental-frequency peak
        f_idx = int(np.argmin(np.abs(r['freqs'] - f0)))
        norm = r['spectrum'][f_idx] + 1e-30
        axes[1].semilogy(r['freqs']/1e6, r['spectrum']/norm,
                         color=colors[label], lw=1.1, label=label)
    axes[1].axvline(f0/1e6, color='gray', ls=':', lw=0.8)
    for n in (2, 3, 4, 5):
        axes[1].axvline(n*f0/1e6, color='gray', ls=':', lw=0.5)
    axes[1].set_xlim(0, min(6*f0, 0.5/dT)/1e6)
    axes[1].set_ylim(1e-6, 3.0)
    axes[1].set(xlabel='Frequency (MHz)', ylabel='|P(f)| / |P(f0)|',
                title='On-axis magnitude spectrum at focus')
    axes[1].legend(); axes[1].grid(True, which='both', alpha=0.3)

    # Peak pressure vs z (PPP and PNP)
    for label, r in results.items():
        axes[2].plot(r['zaxis']*1e3, r['ppp_onaxis']/1e6,
                     color=colors[label], lw=1.2, label=f'{label} PPP')
        axes[2].plot(r['zaxis']*1e3, r['pnp_onaxis']/1e6,
                     color=colors[label], lw=1.0, ls='--',
                     label=f'{label} PNP')
    axes[2].axvline(F_foc*1e3, color='gray', ls=':')
    axes[2].set(xlabel='z (mm)', ylabel='On-axis pressure (MPa)',
                title='Peak positive / negative pressure')
    axes[2].legend(fontsize=8); axes[2].grid(True)

    fig.suptitle(f'Nonlinear shocked focus: p0 = {p0/1e6:.2f} MPa, '
                 f'beta = {beta}, a = 8 lambda, F = 20 mm',
                 fontsize=12)
    plt.tight_layout()
    _save_fig(fig, f'test11d_nonlinear_kfilter{tag}.png')

    np.savez(os.path.join(OUTDIR, f'test11d_nonlinear_kfilter{tag}.npz'),
             zaxis_on=results['k-filter ON']['zaxis'],
             zaxis_off=results['k-filter OFF']['zaxis'],
             ppp_on=results['k-filter ON']['ppp_onaxis'],
             ppp_off=results['k-filter OFF']['ppp_onaxis'],
             pnp_on=results['k-filter ON']['pnp_onaxis'],
             pnp_off=results['k-filter OFF']['pnp_onaxis'],
             waveform_on=results['k-filter ON']['waveform_F'],
             waveform_off=results['k-filter OFF']['waveform_F'],
             spectrum_on=results['k-filter ON']['spectrum'],
             spectrum_off=results['k-filter OFF']['spectrum'],
             freqs=results['k-filter ON']['freqs'],
             taxis=taxis, p0=p0, beta=beta, f0=f0, a=a, F=F_foc)

    # Summary
    r_off = results['k-filter OFF']
    r_on = results['k-filter ON']
    print('\n  ' + '-'*66)
    print(f'  {"config":<16s} {"PPP(F)":>10s} {"PNP(F)":>10s} '
          f'{"max |p|":>10s} {"runtime":>10s}')
    print('  ' + '-'*66)
    for label, r in results.items():
        print(f'  {label:<16s} {r["ppp_onaxis"][r["idx_F"]]/1e6:>9.3f}M '
              f'{r["pnp_onaxis"][r["idx_F"]]/1e6:>9.3f}M '
              f'{r["pmax_onaxis"].max()/1e6:>9.3f}M '
              f'{r["runtime_s"]:>9.1f}s')
    print('  ' + '-'*66)

    return results


if __name__ == '__main__':
    run_nonlinear_shocked_focus()
