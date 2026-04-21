import jax.numpy as jnp
from jax import device_put
from jax.numpy.fft import fftn, ifftn, fftshift, ifftshift, fft, ifft
import jax
import numpy as np
from scipy import signal
import os
import matplotlib.pyplot as plt
from jax import jit, vmap
import time
import pdb
import h5py
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator

from pathlib import Path
from scipy.signal import hilbert
# Enable double precision for JAX
jax.config.update("jax_enable_x64", True)

@jit

def march_step(apaz, HH, abl, afilt3d, N, dZ, dT):
    """Single marching step of the acoustic simulation using JAX."""
    
    # Angular spectrum propagation
    apaz = jnp.real(jnp.fft.ifftn(jnp.fft.ifftshift(jnp.fft.fftshift(jnp.fft.fftn(apaz)) * HH)))
    
    # Apply boundary layer
    apaz *= abl
    
    # Rusanov flux computation
    lambdahalf = jnp.maximum(jnp.abs(apaz[:, :, :-1]), jnp.abs(apaz[:, :, 1:]))
    fluxhalf = -0.5 * (apaz[:, :, :-1]**2 + apaz[:, :, 1:]**2) - lambdahalf * (apaz[:, :, 1:] - apaz[:, :, :-1])
    apaz = apaz.at[:, :, 1:-1].add(-N * dZ / dT * (fluxhalf[:, :, 1:] - fluxhalf[:, :, :-1]))
    
    # Apply attenuation/dispersion
    apaz = jnp.fft.irfft(jnp.fft.rfft(apaz, axis=2) * afilt3d, n=apaz.shape[2], axis=2)
    
    return apaz

def simulate_propagation(apa, prop_dist, dZ, dT, N, HH, abl, afilt3d, basedir, 
                         nX, nY, xaxis, yaxis, t, c0, rho0, pdur, f0):
    """Main simulation loop with visualization"""
    
    apaz = apa.astype(np.float32)
    n_steps = int(np.ceil(prop_dist / dZ))
    pnp, ppp, pI, pax = (np.zeros((nX, nY, n_steps), dtype=np.float32) for _ in range(3)), np.zeros((nT, n_steps), dtype=np.float32)
    (Path(basedir) / 'movies').mkdir(parents=True, exist_ok=True)

    zvec, cc = [dZ], 0
    while sum(zvec) < prop_dist:
        pax[:, cc], pI[:, :, cc] = apaz[nX//2, nY//2, :], np.sum(apaz**2, axis=2)
        pnp[:, :, cc], ppp[:, :, cc] = np.min(apaz, axis=2), np.max(apaz, axis=2)

        if N * dZ/dT * np.max(np.abs(apaz)) > 0.1:
            print('Stability criterion violated, retrying with smaller step size')
            dZ = 0.075 * dT / (np.max(np.abs(apaz)) * N)
            continue  # Recalculate operators outside loop if needed

        zvec.append(dZ)
        print(f'Propagation distance = {sum(zvec):.6f} m')
        apaz = march_step(apaz, HH, abl, afilt3d, N, dZ, dT)
        
        plot_simulation_state(apaz, pI, pnp, cc, zvec, xaxis, yaxis, t, c0, rho0, pdur, f0, Path(basedir) / 'movies')
        cc += 1

    return pnp, ppp, pI, pax

def precalculate_ad_pow2(alpha0: float, nX: int, nY: int, nT: int, dZ: float, dT: float, 
                          c0: float, f0: float, pow: float) -> tuple:
    """
    Calculates attenuation and dispersion filters for wave propagation using a power law.
    
    Parameters:
        alpha0: Base attenuation coefficient.
        nX, nY: Spatial grid dimensions.
        nT: Number of time points.
        dZ: Spatial step size in propagation direction.
        dT: Time step size.
        c0: Reference sound speed.
        f0: Reference frequency.
        pow: Power law exponent for frequency dependence.
        
    Returns:
        afilt3d: 3D complex array containing the attenuation-dispersion filter.
        f: Frequency array.
        attenuation: Attenuation coefficient (Np/m).
        dispersion: Frequency-dependent wave speed (m/s).
    """
    print("Gianmarco Pinton, written on 2017-05-25")
    print("Precalculating attenuation/dispersion filter with power law...")
    
    # Calculate frequency axis (avoid zero frequency)
    f = np.fft.rfftfreq(nT, dT)
    f[0] = f[1] / 2

    # Unit conversion for attenuation and compute frequency-dependent attenuation
    alphaUnitConv = alpha0 / (1e6**pow) * 1e2 / (20 * np.log10(np.e))
    alpha = alphaUnitConv * f**pow

    # Compute dispersion term based on whether the exponent is odd or even
    if pow % 2 == 1:
        alphaStar0 = -2 * alphaUnitConv / ((2 * np.pi)**pow * np.pi) * (np.log(2 * np.pi * f) - np.log(2 * np.pi * f0))
    else:
        alphaStar0 = (alphaUnitConv / ((2 * np.pi)**pow) * np.tan(np.pi * pow / 2) *
                      ((2 * np.pi * f)**(pow - 1) - (2 * np.pi * f0)**(pow - 1)))
    alphaStar = 2 * np.pi * alphaStar0 * f
    dispersion = 1 / ((1 / c0) + (alphaStar / (2 * np.pi * f)))
    
    # Create the complex attenuation-dispersion filter and broadcast to 3D
    afilt = np.exp((-alpha - 1j * alphaStar) * dZ)
    n_freq = nT // 2 + 1
    afilt3d = np.broadcast_to(afilt[None, None, :n_freq], (nX, nY, n_freq))
    
    print("done.")
    return afilt3d, f, alpha, dispersion

def precalculate_mas(nX, nY, nT, dX, dY, dZ, dT, c0):
    """Calculate the modified angular spectrum for wave propagation."""
    print("Gianmarco Pinton, written on 2017-05-25")
    
    # Compute wavenumbers
    kt, kx, ky = [(np.linspace(0, dim-1, dim) / (dim-1) / d * 2 * np.pi - np.mean(np.linspace(0, dim-1, dim) / (dim-1) / d * 2 * np.pi)) 
                  for dim, d in [(nT, dT / c0), (nX, dX), (nY, dY)]]
    
    print("Precalculating modified angular spectrum...")
    
    # Compute wavenumber grid
    kk = kx[:, None]**2 + ky**2
    
    # Compute spectrum
    HH = np.array([
        np.where(kk < k**2, np.exp(dZ * (1j * k - 1j * np.sqrt(k**2 - kk))),
                               np.exp(dZ * (1j * k - np.sqrt(kk - k**2))))
        for k in kt
    ]).transpose(1, 2, 0)
    
    # Zero negative frequencies and double positive ones
    HH[:, :, :(nT+1)//2] = 0
    HH *= 2
    
    print("done.")
    return HH

def ablvec(N: int, n: int) -> np.ndarray:
    """Generates a quadratic absorption profile with full transmission (1) in the middle
    and full absorption (0) at both ends."""
    x = np.linspace(1, 0, n) ** 2
    vec = np.ones(N)
    vec[:n], vec[-n:] = 1 - x, 1 - x[::-1]
    return vec

def precalculate_abl(nX: int, nY: int, nT: int, boundary_factor: float = 0.2) -> np.ndarray:
    """Computes a 3D absorbing boundary layer for wave propagation simulations."""
    if not (0 < boundary_factor < min(nX, nY, nT)):
        raise ValueError('boundary_factor must be between 0 and the minimum of nX, nY, and nT')

    print('Precalculating absorbing boundary layer...')
    
    # Compute boundary widths
    x_bw, y_bw, t_bw = (max(round(dim * boundary_factor), 1) for dim in (nX, nY, nT))

    # Compute boundary profiles
    abl_x, abl_y, abl_t = ablvec(nX, x_bw), ablvec(nY, y_bw), ablvec(nT, t_bw)

    # Compute 3D absorbing layer using broadcasting
    abl = np.outer(abl_x, abl_y).reshape(nX, nY, 1) * abl_t
    
    print('done.')
    return abl

def _get_raw_metadata(file: h5py.File) -> tuple:
    """Get the metadata from a raw file.
    Args:
        file: The data file.

    Returns:
        data: The test data.
        freq_sampling: The time sampling interval (s).
        lats: The lateral transducer element positions (meters).
        els: The elevation transducer elements positions (meters).
        deps: The depths to beamform (meters).
        theta: The transmit angle in theta.
        phi: The transmit angle in phi.
        c0: The speed of sound in tissue.
    """
    fs = file["metadata"]["sequence"]["sample_rate_hz"][()]
    fc = file["metadata"]["sequence"]['transmit_freq_hz'][()]
    # TO DO: add scalar from parameters
    lats = file["metadata"]["transducer"]["lateral"][:].astype(np.float32)
    dx = np.mean(np.diff(lats))
    # lats = np.arange(lats[0], lats[-1], dx / 2).astype(np.float32)
    els = file["metadata"]["transducer"]["elevation"][:].astype(np.float32)
    c0 = file["metadata"]["sequence"]["extras"]["sos_tissue"][()]
    s0, s1 = file["metadata"]["sequence"]["imaging_depth_s"][:]
    d0, d1 = s0 * c0 / 2, s1 * c0 / 2
    deps = np.arange(d0, d1, dx / 2).astype(np.float32)
    # TO DO: this is probably not right
    theta = np.deg2rad(
        file["metadata"]["sequence"]["extras"]["tx_az_angle_deg"][()]
    )
    phi = np.deg2rad(file["metadata"]["sequence"]["extras"]["tx_el_angle_deg"][()])
    receive_delay_s = file["metadata"]["sequence"]['receive_delay_s'][()]
    # assumes plane wave
    transmit_delay_s = np.nanmean(file["metadata"]["sequence"]['transmit_delays_s'][()])
    transmit_delays_s = file["metadata"]["sequence"]['transmit_delays_s'][:]
    transmit_element_mask = file["metadata"]["sequence"]["transmit_element_mask"][:]

    time_offset = receive_delay_s - transmit_delay_s
    return file["data"][:], fs, fc, lats, els, deps, theta, phi, c0, time_offset, transmit_delays_s, transmit_element_mask

def list_h5_data(file: h5py.File):
    """
    Recursively lists all data in an HDF5 file.
    
    Prints:
        - Dataset path.
        - Value and data type if scalar.
        - Shape, dimensions, and data type if array.
    """
    def traverse(group, path=""):
        for key in group:
            item_path = f"{path}/{key}"
            item = group[key]
            if isinstance(item, h5py.Dataset):
                print(f"Dataset: {item_path}")
                if item.shape == ():
                    # Scalar data
                    print(f"  Value: {item[()]}")
                    print(f"  Data Type: {item.dtype}")
                else:
                    # Array data
                    print(f"  Shape: {item.shape}")
                    print(f"  Dimensions: {len(item.shape)}")
                    print(f"  Data Type: {item.dtype}")
            elif isinstance(item, h5py.Group):
                print(f"Group: {item_path}")
                traverse(item, item_path)
            else:
                # For any other type (e.g., h5py.Datatype), skip or print a message.
                print(f"Skipping {item_path} (type {type(item)})")
    
    traverse(file)
def apply_hann_window(ap, fwhm_x=None, fwhm_y=None):
    """Apply a symmetric Hann window to an aperture with optional FWHM scaling."""
    ap_windowed = ap.copy()
    if fwhm_x: ap_windowed *= np.power(np.hanning(ap.shape[0])[:, None], 1/fwhm_x)
    if fwhm_y: ap_windowed *= np.power(np.hanning(ap.shape[1]), 1/fwhm_y)
    return ap_windowed

def create_aperture_mask(nX, nY, xaxis, yaxis, wX, wY, fwhm_x=1, fwhm_y=1):
    """Create a rectangular aperture mask (with Hann windowing) for the given grid.
    
    Parameters:
        nX, nY : int
            Number of grid points in x and y directions.
        xaxis, yaxis : np.ndarray
            1D arrays defining the x and y coordinates.
        wX, wY : float
            Aperture width in the x and y directions.
        fwhm_x, fwhm_y : float, optional
            FWHM parameters for the Hann window along x and y.
            
    Returns:
        np.ndarray
            The windowed aperture mask.
    """
    X, Y = np.meshgrid(xaxis, yaxis, indexing='ij')
    # Create a binary aperture: ones inside the aperture, zeros outside.
    ap = np.where((np.abs(X) <= wX/2) & (np.abs(Y) <= wY/2), 1.0, 0.0)
    # Apply the Hann window with the provided FWHM parameters.
    ap = apply_hann_window(ap, fwhm_x=fwhm_x, fwhm_y=fwhm_y)
    # Ensure that values outside the aperture remain zero.
    ap[(np.abs(X) > wX/2) & (np.abs(Y) > wY/2)] = 0.0
    return ap

def interpolate_3d_xyz(coords, data_3d, xaxis, yaxis, zaxis):
    """Interpolates 3D data on given coordinates."""
    return RegularGridInterpolator((xaxis, yaxis, zaxis), data_3d)(np.asarray(coords, dtype=np.float32))

def remove_mean_delays(delays):
    """Subtract the nanmean from each slice (ignoring NaNs)."""
    delays_zm = np.empty_like(delays)
    for i in range(delays.shape[0]):
        delays_zm[i] = delays[i] - np.nanmean(delays[i])
    return delays_zm

def print_slice_ranges(delays, label):
    """Print the min and max values of each slice with a given label."""
    for i in range(delays.shape[0]):
        print(f"Slice {i+1} {label} - min: {delays[i].min():.2e}, max: {delays[i].max():.2e}")

def interpolate_delays(delays, orig_x, orig_y, new_x, new_y):
    """Interpolate each slice of a 3D delay array onto a new (x,y) grid."""
    num_slices = delays.shape[0]
    interp_delays = np.empty((num_slices, len(new_x), len(new_y)))
    orig_X, orig_Y = np.meshgrid(orig_x, orig_y, indexing='ij')
    new_X, new_Y   = np.meshgrid(new_x, new_y, indexing='ij')
    for i in range(num_slices):
        valid = ~np.isnan(delays[i])
        if np.any(valid):
            valid_points = np.column_stack((orig_X[valid], orig_Y[valid]))
            valid_values = delays[i][valid]
            interp_func = interpolate.LinearNDInterpolator(valid_points, valid_values, fill_value=np.nan)
            slice_interp = interp_func(new_X, new_Y)
            # Replace any remaining NaNs with nearest‐neighbor interpolation.
            if np.any(np.isnan(slice_interp)):
                nn_interp = interpolate.NearestNDInterpolator(valid_points, valid_values)
                nan_mask = np.isnan(slice_interp)
                slice_interp[nan_mask] = nn_interp(new_X[nan_mask], new_Y[nan_mask])
            interp_delays[i] = slice_interp
        else:
            print(f"Warning: Slice {i} contains only NaN values")
            interp_delays[i] = np.nan
    return interp_delays

def create_aperture_mask(nX, nY, xaxis, yaxis, wX, wY):
    """Create a rectangular aperture mask (with Hann windowing) for the given grid."""
    X, Y = np.meshgrid(xaxis, yaxis, indexing='ij')
    # Set ones inside the aperture
    ap = np.where((np.abs(X) <= wX/2) & (np.abs(Y) <= wY/2), 1.0, 0.0)
    # Apply Hann window (assumed to be defined elsewhere)
    ap = apply_hann_window(ap, fwhm_x=0.5, fwhm_y=0.5)
    # Ensure points outside the aperture remain zero
    ap[(np.abs(X) > wX/2) & (np.abs(Y) > wY/2)] = 0.0
    return ap

def create_initial_condition(t, omega0, ncycles, dur, p0):
    """Generate the initial condition (pressure vs time) signal."""
    return np.exp(-(1.05 * t * omega0 / (ncycles * np.pi))**(2 * dur)) * np.sin(t * omega0) * p0

def generate_space_time_field(ap, t, tt, omega0, ncycles, dur, p0):
    """
    Generate the 3D space-time field given an aperture mask, a time axis, and 
    per-pixel time delays (tt). This is vectorized over the spatial grid.
    """
    # Reshape for broadcasting: t becomes (1,1,nT) and tt is (nX, nY, 1)
    t_grid = t[None, None, :]
    tt_grid = tt[..., None]
    field = np.exp(-((1.05 * (t_grid - tt_grid) * omega0 / (ncycles * np.pi))**(2 * dur))) \
            * np.sin(omega0 * (t_grid - tt_grid)) * p0
    return ap[..., None] * field


def plot_propagation_step(apaz, pI, pnp, pax, arrivaltime_matrix, amplitude_mask,
                          xaxis, yaxis, t, zaxis, current_dist, cc,
                          dT, c0, pdur, f0, rho0, save_path):
    """
    Create a multi-panel plot for a propagation step and save the figure.
    
    Modifications:
      - For propagation-dependent plots (subplots 4–11) the entire z-axis is used.
      - The vertical axis (z) is static and flipped (z = 0 at the top) using the input zaxis.
      - The correct aspect ratios are preserved based on the axis units.
    
    Parameters:
      zaxis : 1D numpy array
          The propagation distance axis (e.g. np.arange(n_steps)*dZ).
      current_dist : float
          Current propagation distance (for plot titles).
      cc : int
          The current index (for non-propagation plots).
      (Other parameters are as before.)
    """
    plt.clf()
    plt.figure(figsize=(24, 30))
    
    # Define static vertical (z) limits from the provided zaxis.
    # zaxis[0] is 0 and zaxis[-1] is the maximum propagation distance.
    z_top = zaxis[0]          # should be 0
    z_bottom = zaxis[-1]      # maximum z value
    
    for i in range(12):
        ax = plt.subplot(6, 2, i + 1)
        if i == 0:  # x-t pressure field
            im = ax.imshow(apaz[:, len(yaxis)//2, :].T,
                           extent=[xaxis[0], xaxis[-1], t[0], t[-1]],
                           aspect='auto')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('t (s)')
            ax.set_title(f'Pressure field at z = {current_dist:.3f} m')
            plt.colorbar(im, ax=ax, label='Pa')
        elif i == 1:  # y-t pressure field
            im = ax.imshow(apaz[len(xaxis)//2, :, :].T,
                           extent=[yaxis[0], yaxis[-1], t[0], t[-1]],
                           aspect='auto')
            ax.set_xlabel('y (m)')
            ax.set_ylabel('t (s)')
            ax.set_title(f'Pressure field at z = {current_dist:.3f} m')
            plt.colorbar(im, ax=ax, label='Pa')
        elif i == 2:  # time trace at center
            ax.plot(t, apaz[len(xaxis)//2, len(yaxis)//2, :])
            ax.set_xlabel('t (s)')
            ax.set_ylabel('Pa')
            ax.set_title(f'Center pressure at z = {current_dist:.3f} m')
            ax.grid(True)
        elif i == 3:  # 2D intensity map in x-y plane (from the final z index)
            isppa = pI[:, :, -1] * dT / (c0 * rho0 * pdur) / 10000
            im = ax.imshow(isppa.T,
                           extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
                           aspect='equal')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_title('Isppa (W/cm²)')
            plt.colorbar(im, ax=ax, label='W/cm²')
        elif i in [4, 5]:
            # Intensity profiles along a fixed cross-section over the entire z-axis.
            if i == 4:
                data = pI[:, len(yaxis)//2, :] * dT / (c0 * rho0 * pdur) / 10000
                x_or_y, xlabel = xaxis, 'x (m)'
            else:
                data = pI[len(xaxis)//2, :, :] * dT / (c0 * rho0 * pdur) / 10000
                x_or_y, xlabel = yaxis, 'y (m)'
            im = ax.imshow(data.T,
                           extent=[x_or_y[0], x_or_y[-1], z_bottom, z_top],
                           aspect='auto')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('z (m)')
            ax.set_title('Isppa (W/cm²)')
            ax.set_ylim(z_bottom, z_top)  # static, flipped z-axis
            plt.colorbar(im, ax=ax, label='W/cm²')
        elif i in [6, 7]:
            # MI profiles over the entire z-axis.
            if i == 6:
                mi_data = -pnp[:, len(yaxis)//2, :] / 1e6 / np.sqrt(f0/1e6)
                x_or_y, xlabel = xaxis, 'x (m)'
            else:
                mi_data = -pnp[len(xaxis)//2, :, :] / 1e6 / np.sqrt(f0/1e6)
                x_or_y, xlabel = yaxis, 'y (m)'
            im = ax.imshow(mi_data.T,
                           extent=[x_or_y[0], x_or_y[-1], z_bottom, z_top],
                           aspect='auto')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('z (m)')
            ax.set_title('MI')
            ax.set_ylim(z_bottom, z_top)
            plt.colorbar(im, ax=ax, label='MI')
        elif i in [8, 9]:
            # Arrival time profiles over the entire z-axis.
            if i == 8:
                data = arrivaltime_matrix[:, len(yaxis)//2, :]
                x_or_y, xlabel = xaxis, 'x (m)'
            else:
                data = arrivaltime_matrix[len(xaxis)//2, :, :]
                x_or_y, xlabel = yaxis, 'y (m)'
            im = ax.imshow(data.T,
                           extent=[x_or_y[0], x_or_y[-1], z_bottom, z_top],
                           aspect='auto')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('z (m)')
            ax.set_title('Arrival Times')
            ax.set_ylim(z_bottom, z_top)
            plt.colorbar(im, ax=ax, label='s')
        elif i in [10, 11]:
            # Amplitude mask profiles over the entire z-axis.
            if i == 10:
                data = amplitude_mask[:, len(yaxis)//2, :]
                x_or_y, xlabel = xaxis, 'x (m)'
            else:
                data = amplitude_mask[len(xaxis)//2, :, :]
                x_or_y, xlabel = yaxis, 'y (m)'
            im = ax.imshow(data.T,
                           extent=[x_or_y[0], x_or_y[-1], z_bottom, z_top],
                           aspect='auto')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('z (m)')
            ax.set_title('Amplitude Mask')
            ax.set_ylim(z_bottom, z_top)
            plt.colorbar(im, ax=ax, label='Amplitude')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)

# --- Main Simulation ---
def main():
    print("JAX devices:", jax.devices())
    
    # Load metadata and list file contents.
    file_name = '/home/gfp/Downloads/seq-0-IQ_3D_1cmto6cm_centered_Depth_5MHz_3a/schema/ensemble_2024-12-12T00-20-34-651724.h5'
    with h5py.File(file_name, "r") as f:
        list_h5_data(f)
    with h5py.File(file_name, "r") as f:
        (rf, fs, f0, lats, els, deps, theta, phi, c0, time_offset,
         transmit_delays_s, transmit_element_mask) = _get_raw_metadata(f)
    
    # Remove mean from transmit delays and print range info.
    transmit_delays_s_zm = remove_mean_delays(transmit_delays_s)
    print_slice_ranges(transmit_delays_s, "original")
    print_slice_ranges(transmit_delays_s_zm, "zero-mean")
    
    # --- Physical and Domain Parameters ---
    f0    = 3e6         # Frequency (Hz)
    c0    = 1500        # Speed of sound (m/s)
    rho0  = 1000        # Density (kg/m³)
    array_dx = 2.08e-04
    wX    = 140 * array_dx
    wY    = 64 * array_dx
    prop_dist = 8e-2
    beta  = 3.5
    N     = beta / (2 * c0**3 * rho0)
    omega0 = f0 * 2 * np.pi
    p0    = 0.03e6
    a0    = 0.5
    fcens = np.array([[0, 0, 5e-2]])
    
    # --- Loop Over Focal Configurations ---
    for ff in range(fcens.shape[0]):
        fcen = fcens[ff]
        # Construct output directories.
        fstr = f'/celerina/gfp/mfs/forest/asr3_schema{ff}'.replace('.', 'p').replace('-1', 'w')
        basedir = os.path.join('/celerina/gfp/mfs/angularspectrum_python', fstr)
        os.makedirs(os.path.join(basedir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(basedir, 'movies'), exist_ok=True)
    
        # --- Grid and Wavelength Parameters ---
        lambda_ = c0 / f0
        dX = lambda_ / 5
        dY = dX
        dZ = dX * 8
        dT = dX / (5 * c0)
        bdy = 2e-2
        nX = 2 * round((wX + bdy) / dX / 2) + 1
        nY = 2 * round((wY + bdy) / dX / 2) + 1
        xaxis = np.arange(nX) * dX - (wX/2 + bdy/2)
        yaxis = np.arange(nY) * dX - (wY/2 + bdy/2)
    
        # --- Interpolate Transmit Delays ---
        orig_nx, orig_ny = transmit_delays_s_zm.shape[1:3]
        orig_x = np.linspace(-wX/2, wX/2, orig_nx)
        orig_y = np.linspace(-wY/2, wY/2, orig_ny)
        delays_interp = interpolate_delays(transmit_delays_s_zm, orig_x, orig_y, xaxis, yaxis)
        print(f"Original shape: {transmit_delays_s_zm.shape}, Interpolated shape: {delays_interp.shape}")
        print(f"New grid dimensions: (nX, nY) = ({nX}, {nY})")
    
        # --- Time Axis and Initial Condition ---
        ncycles = 1.5
        dur = 2
        duration = ncycles * 6 * 2 * np.pi / omega0
        nT = round(duration / dT)
        if nT % 2 == 0:
            nT += 1
        t = np.arange(nT) * dT - 2 * ncycles / omega0 * 2 * np.pi
        t -= np.mean(t)
        icvec = create_initial_condition(t, omega0, ncycles, dur, p0)
        plt.figure()
        plt.plot(t, icvec)
        plt.grid(True)
        plt.title('Initial condition (Pa)')
        plt.xlabel('t (s)'); plt.ylabel('Pa')
        plt.savefig(os.path.join(basedir, 'figures', 'icvec.png'), dpi=400)
        plt.close()
    
        pdur = np.sum(np.abs(signal.hilbert(icvec)) > p0/2) * dT
        intensity = np.sum(icvec**2) * dT / (c0 * rho0 * pdur)
        intensity_per_cm2 = intensity / 10000
    
        # --- Aperture Mask ---
        ap = create_aperture_mask(nX, nY, xaxis, yaxis, wX, wY)
        plt.figure()
        plt.imshow(ap.T, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]], aspect='equal')
        plt.title('Aperture mask')
        plt.xlabel('x (m)'); plt.ylabel('y (m)')
        plt.savefig(os.path.join(basedir, 'figures', 'ic_mask.png'), dpi=400)
        plt.close()
    
        # --- Space-Time Field ---
        # Use the first slice of interpolated delays as the delay map.
        tt = delays_interp[0]
        tt[np.isnan(tt)] = 0
        tt = tt - np.mean(tt[tt != 0])
        apa = generate_space_time_field(ap, t, tt, omega0, ncycles, dur, p0)
    
        plt.figure()
        plt.imshow(apa[:, :, nT//2].T, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]], aspect='equal')
        plt.title('Initial condition (Pa) - x-y slice')
        plt.xlabel('x (m)'); plt.ylabel('y (m)')
        plt.colorbar(label='Pa')
        plt.savefig(os.path.join(basedir, 'figures', 'ic_xy.png'), dpi=400)
        plt.close()
    
        taxis = np.arange(nT) * dT
        HH = precalculate_mas(nX, nY, nT, dX, dY, dZ, dT, c0).astype(np.complex64)
        abl = precalculate_abl(nX, nY, nT, 2e-3/nY/dY).astype(np.float32)
        plt.figure()
        plt.imshow(abl[:, nY//2, :].T, aspect='auto')
        plt.colorbar(label='Absorption Coefficient')
        plt.title('Absorbing Boundary Layer Profile')
        plt.xlabel('X Position'); plt.ylabel('Time')
        plt.savefig(os.path.join(basedir, 'figures', 'abl.png'), dpi=400)
        plt.close()
    
        # --- Absorption/Dispersion Filter ---
        if a0 == -1:  # Water
            alpha0 = 2.17e-3
            afilt3d = precalculate_ad(alpha0, nX, nY, nT, dZ, dT)
            pow_val = 2
        else:
            alpha0 = a0
            pow_val = 1
            afilt3d, _, _, _ = precalculate_ad_pow2(alpha0, nX, nY, nT, dZ, dT, c0, omega0/(2*np.pi), pow_val)
    
        HH = jnp.array(HH, dtype=jnp.complex64)
        abl = jnp.array(abl, dtype=jnp.float32)
        afilt3d = jnp.array(afilt3d * 0 + 1, dtype=jnp.complex64)
    
        # --- Propagation Arrays ---
        n_steps = int(np.ceil(prop_dist / dZ))
        pnp = np.zeros((nX, nY, n_steps), dtype=np.float32)
        ppp = np.zeros((nX, nY, n_steps), dtype=np.float32)
        pI  = np.zeros((nX, nY, n_steps), dtype=np.float32)
        pax = np.zeros((nT, n_steps), dtype=np.float32)
        arrivaltime_matrix = np.zeros((nX, nY, n_steps), dtype=np.float32)
        amplitude_mask = np.zeros((nX, nY, n_steps), dtype=np.float32)
        zaxis = np.arange(n_steps) * dZ

        # --- Propagation Loop ---
        apaz = jnp.array(apa, dtype=jnp.float32)
        zvec = [0]
        cc = 0
        fig_dir = os.path.join(basedir, 'movies')
        start_time = time.time()
    
        while sum(zvec) < prop_dist:
            pax[:, cc] = apaz[nX//2, nY//2, :]
            pI[:, :, cc]  = np.sum(apaz**2, axis=2)
            pnp[:, :, cc] = np.min(apaz, axis=2)
            ppp[:, :, cc] = np.max(apaz, axis=2)
    
            peak_idxs = np.argmax(apaz, axis=2)
            arrivaltime_matrix[:, :, cc] = taxis[peak_idxs] + cc * dZ / c0
            pIslice = np.sum(apaz**2, axis=2)
            amplitude_mask[:, :, cc] = np.where(np.abs(pIslice) > 0.1 * np.mean(pIslice), 1, 0)
    
            # Stability check – if violated, recalc with a smaller step size.
            if N * dZ / dT * np.max(np.abs(apaz)) > 0.1:
                print('Stability criterion violated, retrying with smaller step size')
                dZ = 0.075 * dT / (np.max(np.abs(apaz)) * N)
                n_steps = int(np.ceil(prop_dist / dZ))
                zaxis = np.arange(n_steps) * dZ
                cc = 0
                zvec = [0]
                HH = precalculate_mas(nX, nY, nT, dX, dY, dZ, dT, c0).astype(np.complex64)
                abl = precalculate_abl(nX, nY, nT, 2e-3/nY/dY).astype(np.float32)
                if a0 == -1:
                    afilt3d = precalculate_ad(alpha0, nX, nY, nT, dZ, dT)
                else:
                    afilt3d, _, _, _ = precalculate_ad_pow2(alpha0, nX, nY, nT, dZ, dT, c0, omega0/(2*np.pi), pow_val)
                HH = jnp.array(HH, dtype=jnp.complex64)
                abl = jnp.array(abl, dtype=jnp.float32)
                afilt3d = jnp.array(afilt3d * 0 + 1, dtype=jnp.complex64)
                continue
    
            current_dist = sum(zvec)
            print(f'Propagation distance = {current_dist:.6f} m')
    
            # One propagation step (using JAX)
            apaz = march_step(apaz, HH, abl, afilt3d, N, dZ, dT)
    
            # Plot and save the propagation state.
            plot_path = os.path.join(fig_dir, f'agp_{cc:04d}.jpg')
            plot_propagation_step(apaz, pI, pnp, pax, arrivaltime_matrix, amplitude_mask,
                                  xaxis, yaxis, t, zaxis, current_dist, cc,
                                  dT, c0, pdur, f0, rho0, plot_path)
            zvec.append(dZ)
            cc += 1
    
        elapsed_time = time.time() - start_time
        print(f"Simulation run time: {elapsed_time:.2f} seconds")
    
        # Save final results.
        np.savez(os.path.join(basedir, 'results.npz'),
                 pnp=pnp[:, :, :cc], ppp=ppp[:, :, :cc],
                 pI=pI[:, :, :cc], pax=pax[:, :cc],
                 x=xaxis, y=yaxis, t=t, z=np.cumsum(zvec[:cc]))
    
        print("Simulation completed successfully.")

if __name__ == "__main__":
    main()