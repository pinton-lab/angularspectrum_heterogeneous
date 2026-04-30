"""
DC-bin / nT invariance test for precalculate_mas.

Regression test for the analytic-signal slice fix:
    HH[:, :, :nT // 2 + 1] = 0   # was (nT + 1) // 2 — left DC un-zeroed for even nT
    HH *= 2
The buggy version doubled the DC bin once per propagation step, producing
2^N runaway for any source field with a finite DC residue.

Test: drive a uniform (in x, y) sine pulse, propagate a fixed distance over a
few even nT, and check the peak field magnitude is independent of nT. Without
the fix, peak |p| diverges as nT grows because every even nT exposes the bug
and more steps means more 2x amplification.

Run:
    python validate_dc_invariance.py
Exits non-zero if the across-nT spread exceeds tolerance.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from angular_spectrum_solver import precalculate_mas


def _propagate_uniform(nT, nX=32, nY=32, f0=1.0e6, c0=1500.0, n_cycles=3,
                       n_steps=64, dz_per_lambda=0.25):
    """Uniform plane-wave-ish source in (x, y), propagate n_steps in water."""
    lam = c0 / f0
    dX = lam / 8
    dY = lam / 8
    dZ = lam * dz_per_lambda
    dT = 1.0 / (f0 * 16)

    t = np.arange(nT) * dT
    env_center = n_cycles / f0 * 0.5
    env = np.exp(-((t - env_center) / (n_cycles / f0 / 4)) ** 2)
    pulse = (np.sin(2 * np.pi * f0 * t) * env).astype(np.float32)

    field = np.broadcast_to(pulse[None, None, :], (nX, nY, nT)).astype(np.float32).copy()

    HH, _ = precalculate_mas(nX, nY, nT, dX, dY, dZ, dT, c0,
                             split_step=False, adaptive_filtering=False)
    HH = HH.astype(np.complex64)

    src_dc = np.abs(np.fft.fft(field, axis=2)[..., 0]).max()

    for _ in range(n_steps):
        F = np.fft.fftn(field)
        F = np.fft.fftshift(F) * HH
        F = np.fft.ifftshift(F)
        field = np.real(np.fft.ifftn(F)).astype(np.float32)

    peak = float(np.max(np.abs(field)))
    final_dc = float(np.abs(np.fft.fft(field, axis=2)[..., 0]).max())
    return peak, src_dc, final_dc


def main():
    nT_values = [120, 240, 480]
    n_steps = 64
    print(f'DC-bin invariance test — nT in {nT_values}, {n_steps} steps, water\n')

    peaks = {}
    dc_growth = {}
    for nT in nT_values:
        peak, src_dc, final_dc = _propagate_uniform(nT, n_steps=n_steps)
        peaks[nT] = peak
        dc_growth[nT] = final_dc / max(src_dc, 1e-30)
        print(f'  nT={nT:>4d}  peak|p|={peak:.4e}  src DC={src_dc:.2e}  '
              f'final DC={final_dc:.2e}  ratio={dc_growth[nT]:.2e}')

    # Primary regression check: DC component must NOT grow. With the bug,
    # ratio would be ~2**n_steps; with the fix it's <= 1 (and typically
    # many orders of magnitude smaller because the propagator zeros DC).
    max_ratio = max(dc_growth.values())
    pmin = min(peaks.values())
    pmax = max(peaks.values())
    spread = (pmax - pmin) / pmin
    print(f'\n  max DC growth ratio = {max_ratio:.2e}  (bug would give ~{2.0**n_steps:.1e})')
    print(f'  peak spread (max-min)/min = {spread*100:.3f} %')

    fail = False
    if max_ratio > 1.0:
        print(f'\nFAIL: DC bin grew (ratio {max_ratio:.2e} > 1).')
        print('  precalculate_mas DC-zeroing slice has regressed.')
        fail = True
    if spread > 0.05:
        print(f'\nFAIL: peak spread {spread*100:.2f}% > 5% — propagator inconsistent across nT.')
        fail = True
    if fail:
        sys.exit(1)
    print('\nPASS: DC bin is suppressed and peak |p| is consistent across nT.')


if __name__ == '__main__':
    main()
