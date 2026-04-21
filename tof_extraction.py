"""Time-of-flight extraction from a pressure field.

Two estimators live here. Both take a field of shape ``(nX, nY, nT)`` and
return per-(x, y) absolute time-of-flight (pulse time + z_cumulative/c0).

``extract_tof_envelope`` — Hilbert envelope argmax (``ratio=1.0``) or
leading-edge threshold (``ratio<1``, e.g. 0.1 for 10 % of peak). Fast,
template-free, and accuracy is capped by the sample period because the
result is locked to the discrete time grid. Leading-edge variants also
carry a systematic offset relative to the envelope peak (roughly half
the pulse duration), which must be corrected externally if the two are
to be mixed.

``extract_tof_matched_filter_parabolic`` — FFT cross-correlation of each
trace against a known reference waveform, followed by parabolic
refinement of the correlation peak for sub-sample accuracy. Needs a
reference pulse and its envelope-peak time. Works well when the received
pulse still resembles the reference; performance degrades as waveform
distortion (e.g. from strong aberration or dispersion) accumulates.

All functions accept numpy, cupy, or JAX arrays transparently. Work is
done on the host via numpy / scipy; device arrays are moved to host with
``np.asarray``. This keeps the implementation simple and is appropriate
for typical ASR grids where host transfer cost is small relative to the
JAX march step.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import hilbert


def extract_tof_envelope(field, t, z_cumulative: float, c0: float,
                         ratio: float = 1.0) -> np.ndarray:
    """Return per-(x, y) time-of-flight in seconds.

    Parameters
    ----------
    field : array of shape (nX, nY, nT)
        Real-valued pressure field. numpy, cupy, or JAX arrays are
        accepted; all are converted to numpy for the Hilbert transform.
    t : array of shape (nT,)
        Time axis (s).
    z_cumulative : float
        Cumulative propagation distance at the current step (m).
    c0 : float
        Reference sound speed (m/s) used to add the geometric offset.
    ratio : float, default 1.0
        Threshold fraction of the envelope peak. ``1.0`` picks the peak
        time; smaller values pick earlier samples on the leading edge.

    Returns
    -------
    tof : ndarray (nX, nY) float32
        Time-of-flight = (first index where env >= ratio * peak) * dT
        + z_cumulative / c0.
    """
    field_np = np.asarray(field)
    nX, nY, nT = field_np.shape
    X = field_np.reshape(-1, nT).T                       # (nT, nX*nY)
    env = np.abs(hilbert(X.astype(np.float64), axis=0)).astype(np.float32)

    peak = env.max(axis=0)                                # (nX*nY,)
    thr = float(ratio) * peak
    above = env >= thr[np.newaxis, :]

    # argmax returns the first True index along axis 0 (guaranteed to
    # exist when ratio <= 1 and peak > 0; for silent columns all False,
    # argmax returns 0 which we mask to NaN below).
    first_idx = np.argmax(above, axis=0)
    silent = peak <= 0
    t_np = np.asarray(t)
    tof_flat = t_np[first_idx] + z_cumulative / c0
    tof_flat = tof_flat.astype(np.float32)
    if silent.any():
        tof_flat[silent] = np.nan
    return tof_flat.reshape(nX, nY)


def extract_tof_matched_filter_parabolic(field, t, z_cumulative: float,
                                         c0: float,
                                         ref_trace: np.ndarray,
                                         t_ref_peak_s: float) -> np.ndarray:
    """Sub-sample TOF via FFT cross-correlation with a reference pulse
    + parabolic refinement of the correlation peak.

    For each (x, y) trace we compute the circular cross-correlation
    with ``ref_trace`` using FFTs. The integer lag at the xcorr
    maximum is refined to sub-sample precision by fitting a parabola
    through the three samples centred on the peak. Time-of-flight at
    the voxel is ``t_ref_peak_s + lag * dT + z_cumulative / c0``.

    Typically more accurate than ``extract_tof_envelope`` on the same
    data because the parabolic fit removes the sample-grid quantization
    and the correlation integrates over the whole pulse rather than
    relying on a single point of the envelope. See the comparison
    notebook at
    ``tests/figure_tof_methods_aberrated_planewave_100ns_r5.py`` for
    a worked example.

    Parameters
    ----------
    field : array of shape (nX, nY, nT)
        Real-valued pressure field. numpy, cupy, or JAX arrays are
        accepted; all are converted to numpy.
    t : array of shape (nT,)
        Time axis (s).
    z_cumulative : float
        Cumulative propagation distance at the current step (m).
    c0 : float
        Reference sound speed (m/s) used to add the geometric offset.
    ref_trace : array of shape (nT,)
        Reference waveform — typically the source pulse or a clean
        central-pixel trace. Must be the same length as ``field``
        along the time axis.
    t_ref_peak_s : float
        Known time (s) at which ``ref_trace`` itself peaks (i.e.,
        the envelope-argmax time of the reference). Used to absolute-
        anchor the returned TOF.

    Returns
    -------
    tof : ndarray (nX, nY) float32
        Time-of-flight, in seconds, with sub-sample accuracy.
    """
    field_np = np.asarray(field)
    nX, nY, nT = field_np.shape
    dT = float(np.asarray(t)[1] - np.asarray(t)[0])

    F = np.fft.fft(field_np.astype(np.float64), axis=-1)
    R = np.conj(np.fft.fft(np.asarray(ref_trace).astype(np.float64), n=nT))
    xcorr = np.real(np.fft.ifft(F * R[None, None, :], axis=-1))
    flat = xcorr.reshape(-1, nT)                           # (N, nT)

    idx = np.argmax(flat, axis=1)
    cols = np.arange(flat.shape[0])
    idx_int = np.clip(idx, 1, nT - 2)
    y0 = flat[cols, idx_int - 1]
    y1 = flat[cols, idx_int]
    y2 = flat[cols, idx_int + 1]
    denom = y0 - 2 * y1 + y2
    delta = np.where(np.abs(denom) > 1e-12,
                     0.5 * (y0 - y2) / denom, 0.0)
    # don't refine at the time-axis boundaries
    delta = np.where((idx == 0) | (idx == nT - 1), 0.0, delta)

    lag = idx + delta
    # wrap negative lags (circular xcorr)
    lag = np.where(lag > nT // 2, lag - nT, lag)

    tof_flat = (float(t_ref_peak_s) + lag * dT
                + z_cumulative / c0).astype(np.float32)
    return tof_flat.reshape(nX, nY)
