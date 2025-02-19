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


import sys

# Correctly add the path to the beamforming3d_forest folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'beamforming3d_forest')))
#In interactive mode, manually specify the path to the target directory with 
#sys.path.append(os.path.abspath('../beamforming3d_forest'))
# Now you can import the functions
from beamform import iq_beamform, rf_beamform

# Enable double precision for JAX
jax.config.update("jax_enable_x64", True)

@jit
def march_step(apaz, HH, abl, afilt3d, N, dZ, dT):
    """
    Single marching step of the acoustic simulation using JAX.
    
    This function implements:
    1. Angular spectrum propagation
    2. Boundary layer absorption
    3. Rusanov flux computation
    4. Attenuation/dispersion
    """
      # Angular Spectrum propagation (using complex FFT)
    apaz = jnp.real(jnp.fft.ifftn(
        jnp.fft.ifftshift(
            jnp.fft.fftshift(jnp.fft.fftn(apaz)) * HH
        )
    ))
    
    # Apply boundary layer
    apaz = apaz * abl
       # Rusanov flux computation
    lambdahalf = jnp.maximum(
        jnp.abs(apaz[:,:,:-1]),
        jnp.abs(apaz[:,:,1:])
    )
    
    fluxhalf = -(apaz[:,:,:-1]**2 + apaz[:,:,1:]**2)/2 - \
               lambdahalf * (apaz[:,:,1:] - apaz[:,:,:-1])
    
    flux_diff = fluxhalf[:,:,1:] - fluxhalf[:,:,:-1]
    apaz = apaz.at[:,:,1:-1].add(-N * dZ / dT * flux_diff)
    
    # Apply attenuation/dispersion using real FFT
    apaz = jnp.fft.irfft(
        jnp.fft.rfft(apaz, axis=2) * afilt3d,
        n=apaz.shape[2],
        axis=2
    )
    
    return apaz

def simulate_propagation(apa, prop_dist, dZ, dT, N, HH, abl, afilt3d, basedir, 
                        nX, nY, xaxis, yaxis, t, c0, rho0, pdur, f0):
    """Main simulation loop with visualization"""
    apaz = apa.astype(np.float32)
    zvec = [dZ]
    
    # Initialize arrays for storing results
    n_steps = int(np.ceil(prop_dist/dZ))
    pnp = np.zeros((nX, nY, n_steps), dtype=np.float32)
    ppp = np.zeros((nX, nY, n_steps), dtype=np.float32)
    pI = np.zeros((nX, nY, n_steps), dtype=np.float32)
    pax = np.zeros((nT, n_steps), dtype=np.float32)
    
    # Create movie directory
    movie_dir = Path(basedir) / 'movies'
    movie_dir.mkdir(parents=True, exist_ok=True)
    
    cc = 0
    while sum(zvec) < prop_dist:
        # Store current state
        pax[:, cc] = apaz[nX//2, nY//2, :]
        pI[:, :, cc] = np.sum(apaz**2, axis=2)
        pnp[:, :, cc] = np.min(apaz, axis=2)
        ppp[:, :, cc] = np.max(apaz, axis=2)
        
        # Check stability criterion
        if N * dZ/dT * np.max(np.abs(apaz)) > 0.1:
            print('Stability criterion violated, retrying with smaller step size')
            dZ = 0.075 * dT / (np.max(np.abs(apaz)) * N)
            # Recalculate all operators with new dZ
            # [Code to recalculate HH, abl, afilt3d goes here]
            continue
            
        zvec.append(dZ)
        print(f'Propagation distance = {sum(zvec):.6f} m')
        
        # Perform one step of propagation
        apaz = march_step(apaz, HH, abl, afilt3d, N, dZ, dT)
        
        # Visualization code
        plot_simulation_state(apaz, pI, pnp, cc, zvec, xaxis, yaxis, t,
                            c0, rho0, pdur, f0, movie_dir)
        
        cc += 1
    
    return pnp, ppp, pI, pax

def save_fig(fig, path):
    """Helper function to save figures, similar to MATLAB's saveFig"""
    fig.savefig(path)
    plt.close(fig)

def plot_simulation_state(apaz, pI, pnp, cc, zvec, xaxis, yaxis, t,
                         c0, rho0, pdur, f0, movie_dir):
    """Creates visualization plots for the current simulation state"""
    fig, axs = plt.subplots(4, 2, figsize=(20, 24))
    
    # Time-space plots
    idx, idy = len(xaxis)//2, len(yaxis)//2
    
    # Pressure field plots
    im = axs[0,0].imshow(apaz[:,idy,:].T, extent=[t[0], t[-1], xaxis[0], xaxis[-1]], 
                         aspect='auto')
    plt.colorbar(im, ax=axs[0,0])
    axs[0,0].set(xlabel='t (s)', ylabel='x (m)', 
                 title=f'Pressure field at {sum(zvec):.3f} m')
    
    im = axs[0,1].imshow(apaz[idx,:,:].T, extent=[t[0], t[-1], yaxis[0], yaxis[-1]], 
                         aspect='auto')
    plt.colorbar(im, ax=axs[0,1])
    axs[0,1].set(xlabel='t (s)', ylabel='y (m)', 
                 title=f'Pressure field at {sum(zvec):.3f} m')
    
    # Time trace
    axs[1,0].plot(t, apaz[idx,idy,:])
    axs[1,0].grid(True)
    axs[1,0].set(xlabel='t (s)', ylabel='Pa', 
                 title=f'Pressure at {sum(zvec):.3f} m')
    
    # Intensity plots
    isppa = pI[:,:,cc] * dT/(c0*rho0*pdur)/10000
    im = axs[1,1].imshow(isppa.T, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]], 
                         aspect='equal')
    plt.colorbar(im, ax=axs[1,1], label='W/cm²')
    axs[1,1].set(xlabel='x (m)', ylabel='y (m)', title='Isppa')
    
    # Additional plots...
    # [Code for remaining subplots goes here]
    
    plt.tight_layout()
    plt.savefig(movie_dir / f'agp_{cc:04d}.jpg')
    plt.close()

def precalculate_ad(alpha0: float, nX: int, nY: int, nT: int, dZ: float, dT: float) -> np.ndarray:
    """
    Calculates a 3D attenuation filter for wave propagation using a quadratic frequency dependence.
    This function models frequency-dependent attenuation in materials like water where
    attenuation increases with the square of frequency.
    
    Parameters:
        alpha0: Base attenuation coefficient
        nX, nY: Spatial grid dimensions
        nT: Number of time points
        dZ: Spatial step size in propagation direction
        dT: Time step size
        
    Returns:
        afilt3d: 3D complex array containing the attenuation filter
    """
    print("Gianmarco Pinton, written on 2017-05-25")
    print("Precalculating attenuation/dispersion filter...")
    
    # Calculate frequency array
    # The frequencies range from 0 to (nT-1)/(nT*dT) = 1/dT
    f = np.arange(nT) / (nT * dT)
    
    # Calculate frequency-dependent attenuation coefficient
    # Converting from dB/cm to Nepers/m with appropriate scaling
    alpha = alpha0 / 1e12 * 1e2 / (20 * np.log10(np.e)) * f**2
    
    # Create the attenuation filter
    # exp(-α*dz) gives the amplitude reduction over distance dZ
    afilt = np.exp(-alpha * dZ)
    
    # Prevent numerical instabilities by clipping negative values
    afilt = np.clip(afilt, 0, None)
    
    # Create 3D filter by repeating the 1D filter across spatial dimensions
    # Using broadcasting instead of loops for better efficiency
    afilt3d = np.broadcast_to(afilt[np.newaxis, np.newaxis, :], (nX, nY, nT))
    
    print("done.")
    return afilt3d

def precalculate_ad_pow2(alpha0: float, nX: int, nY: int, nT: int, dZ: float, dT: float, 
                        c0: float, f0: float, pow: float) -> tuple:
    """
    Calculates attenuation and dispersion filters for wave propagation using a power law.
    This more general model accounts for both amplitude attenuation and phase velocity
    dispersion, which are linked through the Kramers-Kronig relations.
    
    Parameters:
        alpha0: Base attenuation coefficient
        nX, nY: Spatial grid dimensions
        nT: Number of time points
        dZ: Spatial step size in propagation direction
        dT: Time step size
        c0: Reference sound speed
        f0: Reference frequency
        pow: Power law exponent for frequency dependence
        
    Returns:
        afilt3d: 3D complex array containing the attenuation-dispersion filter
        f: Frequency array
        attenuation: Attenuation coefficient (Np/m)
        dispersion: Frequency-dependent wave speed (m/s)
    """
    print("Gianmarco Pinton, written on 2017-05-25")
    print("Precalculating attenuation/dispersion filter with power law...")
    
    # Calculate frequencies, avoiding division by zero
    f = np.fft.rfftfreq(nT, dT)  # Use rfftfreq to match the real FFT
    f[0] = f[1]/2  # Handle zero frequency
    
    # Unit conversion for attenuation
    alphaUnitConv = alpha0 / (1e6**pow) * 1e2 / (20 * np.log10(np.e))
    alpha = alphaUnitConv * f**pow
    
    # Calculate dispersion based on power law
    if pow % 2 == 1:
        alphaStar0 = (-2 * alphaUnitConv / ((2 * np.pi)**pow) / np.pi) * \
                     (np.log(2 * np.pi * f) - np.log(2 * np.pi * f0))
    else:
        alphaStar0 = (alphaUnitConv / (2 * np.pi)**pow) * np.tan(np.pi * pow / 2) * \
                     ((2 * np.pi * f)**(pow - 1) - (2 * np.pi * f0)**(pow - 1))
    
    alphaStar = 2 * np.pi * alphaStar0 * f
    dispersion = 1 / ((1/c0) + (alphaStar/2/np.pi/f))
    
    # Create filter
    afilt = np.exp((-alpha - 1j * alphaStar) * dZ)
    
    # Broadcast to match rfft output shape
    n_freq = nT//2 + 1
    afilt3d = np.broadcast_to(afilt[np.newaxis, np.newaxis, :n_freq], 
                             (nX, nY, n_freq))
    
    print("done.")
    return afilt3d, f, alpha, dispersion

def precalculate_mas(nX, nY, nT, dX, dY, dZ, dT, c0):
    """Calculate the modified angular spectrum for wave propagation."""
    print("Gianmarco Pinton, written on 2017-05-25")
    
    # Match MATLAB's wavenumber calculation exactly
    kt = np.linspace(0, nT-1, nT) / (nT-1) / dT * 2 * np.pi / c0
    kt = kt - np.mean(kt)
    
    kx = np.linspace(0, nX-1, nX) / (nX-1) / dX * 2 * np.pi
    kx = kx - np.mean(kx)
    
    ky = np.linspace(0, nY-1, nY) / (nY-1) / dY * 2 * np.pi
    ky = ky - np.mean(ky)
    
    # Initialize output array
    HH = np.zeros((nX, nY, nT), dtype=np.complex128)
    
    print("Precalculating modified angular spectrum...")
    
    # Calculate wavenumber grid once
    kk = np.zeros((nX, nY))
    for ii in range(nX):
        for jj in range(nY):
            kk[ii,jj] = kx[ii]**2 + ky[jj]**2
    
    # Loop through temporal frequencies
    for m in range(nT):
        k = kt[m]
        
        # Calculate propagating and evanescent wave components
        H2 = np.exp(dZ * (1j * k - np.sqrt(kk - k**2)))
        H1 = np.exp(dZ * (1j * k - 1j * np.sqrt(k**2 - kk)))
        
        # Combine using MATLAB's approach
        H = H2.copy()
        mask = kk < k**2
        H[mask] = H1[mask]
        
        HH[:,:,m] = H
    
    # Zero out negative frequencies and double positive ones
    HH[:,:,:(nT+1)//2] = 0
    HH = HH * 2
    
    print("done.")
    return HH

def ablvec(N: int, n: int) -> np.ndarray:
    """
    Creates a vector with quadratic absorption profiles at both ends.
    This function generates a vector that has maximum transmission (1.0) in the middle
    and gradually decreases to minimum transmission (0.0) at both ends using a quadratic profile.
    
    Parameters:
        N: Total length of the vector
        n: Width of the absorption region at each end
        
    Returns:
        vec: Array of shape (N,) containing the absorption profile
             Values range from 0 (full absorption) to 1 (no absorption)
    """
    # Initialize vector with zeros
    vec = np.zeros(N)
    
    # Create absorption profile at the start of the vector
    for nn in range(n):
        # Calculate normalized position from the edge (1 at edge, 0 at n points in)
        x = (n - nn - 1) / n
        # Apply quadratic profile
        vec[nn] = x**2
        
    # Create absorption profile at the end of the vector
    for nn in range(N - n, N):
        # Calculate normalized position from the inner edge (0 at inner edge, 1 at boundary)
        x = (nn - (N - n - 1)) / n
        # Apply quadratic profile
        vec[nn] = x**2
    
    # Invert the profile so 1 represents full transmission and 0 represents full absorption
    vec = 1 - vec
    
    return vec


def precalculate_abl(nX: int, nY: int, nT: int, boundary_factor: float = 0.2) -> np.ndarray:
    """
    Calculates a 3D absorbing boundary layer array used in wave propagation simulations.
    The boundary layer helps prevent reflections at the computational domain boundaries.
    
    Parameters:
        nX: Number of points in X dimension
        nY: Number of points in Y dimension
        nT: Number of points in time dimension
        boundary_factor: Fraction of domain size to use for boundary layer (default: 1/5)
        
    Returns:
        abl: 3D numpy array containing the absorbing boundary layer values
             Shape: (nX, nY, nT), dtype: np.float32
    """
    # Input validation
    if boundary_factor <= 0 or boundary_factor >= min(nX, nY, nT):
        raise ValueError('boundary_factor must be greater than 0 and less than the minimum of nX, nY, and nT')
    
    print('Precalculating absorbing boundary layer...')
    
    # Calculate boundary widths for each dimension
    # max() ensures we have at least 1 point in the boundary layer
    x_boundary_width = max(round(nX * boundary_factor), 1)
    y_boundary_width = max(round(nY * boundary_factor), 1)
    t_boundary_width = max(round(nT * boundary_factor), 1)
    
    # Calculate 2D spatial boundary layer using outer product of X and Y vectors
    abl_tmp = np.outer(ablvec(nX, x_boundary_width), 
                      ablvec(nY, y_boundary_width))
    
    # Calculate temporal boundary vector
    abl_vec = ablvec(nT, t_boundary_width)
    
    # Initialize the 3D array with zeros
    abl = np.zeros((nX, nY, nT), dtype=np.float32)
    
    # Fill the 3D array by multiplying the 2D spatial layer with each time point
    # This is equivalent to the MATLAB loop but uses broadcasting
    abl = abl_tmp[:, :, np.newaxis] * abl_vec[np.newaxis, np.newaxis, :]
    
    print('done.')
    return abl

def _get_raw_metadata(file: h5py.File) -> tuple:
    """
    Get the metadata from a raw file in a robust way.
    
    If certain expected keys are missing (e.g. in "extras"), default values are used.
    
    Returns:
        data: The test data (numpy array).
        fs: Sampling rate.
        fc: Transmit frequency.
        lats: Lateral transducer element positions (np.float32).
        els: Elevation transducer element positions (np.float32).
        deps: Depths to beamform (np.float32).
        theta: Transmit azimuth angle in radians.
        phi: Transmit elevation angle in radians.
        c0: Speed of sound (default 1540 m/s if not provided).
        time_offset: receive_delay_s minus transmit_delay_s.
        transmit_delays_s: Array of transmit delays.
        transmit_element_mask: Transmit element mask.
    """
    def safe_access(keys, default=None, slice_all=False):
        """Traverse nested keys in the file; if any key is missing, return default."""
        try:
            obj = file
            for k in keys:
                obj = obj[k]
            return obj[:] if slice_all else obj[()]
        except KeyError:
            return default

    # Required parameters (raise error if missing)
    fs = safe_access(["metadata", "sequence", "sample_rate_hz"])
    if fs is None:
        raise ValueError("Missing sample_rate_hz in metadata")
    fc = safe_access(["metadata", "sequence", "transmit_freq_hz"])
    if fc is None:
        raise ValueError("Missing transmit_freq_hz in metadata")
    
    lats = safe_access(["metadata", "transducer", "lateral"], slice_all=True)
    if lats is None:
        raise ValueError("Missing lateral transducer positions")
    lats = lats.astype(np.float32)
    dx = np.mean(np.diff(lats))
    
    els = safe_access(["metadata", "transducer", "elevation"], slice_all=True)
    if els is None:
        raise ValueError("Missing elevation transducer positions")
    els = els.astype(np.float32)
    
    # For keys under "extras", use defaults if missing.
    c0 = safe_access(["metadata", "sequence", "extras", "sos_tissue"])
    c0 = c0 if c0 is not None else 1540.0
    theta_deg = safe_access(["metadata", "sequence", "extras", "tx_az_angle_deg"])
    theta = np.deg2rad(theta_deg) if theta_deg is not None else 0.0
    phi_deg = safe_access(["metadata", "sequence", "extras", "tx_el_angle_deg"])
    phi = np.deg2rad(phi_deg) if phi_deg is not None else 0.0

    s_vals = safe_access(["metadata", "sequence", "imaging_depth_s"], slice_all=True)
    if s_vals is None or len(s_vals) < 2:
        raise ValueError("Missing or incomplete imaging_depth_s in metadata")
    s0, s1 = s_vals
    d0, d1 = s0 * c0 / 2, s1 * c0 / 2
    deps = np.arange(d0, d1, dx / 2).astype(np.float32)
    
    receive_delay_s = safe_access(["metadata", "sequence", "receive_delay_s"], default=0.0)
    
    transmit_delays_s_arr = safe_access(["metadata", "sequence", "transmit_delays_s"], slice_all=True)
    if transmit_delays_s_arr is None or len(transmit_delays_s_arr) == 0:
        transmit_delays_s_arr = np.array([], dtype=np.float32)
        transmit_delay_s = 0.0
    else:
        transmit_delay_s = np.nanmean(transmit_delays_s_arr)
    
    transmit_element_mask = safe_access(["metadata", "sequence", "transmit_element_mask"], slice_all=True)
    # Compute time offset.
    time_offset = receive_delay_s - transmit_delay_s
    
    data = safe_access(["data"], slice_all=True)
    if data is None:
        raise ValueError("Missing data in file")
    
    return (data, fs, fc, lats, els, deps, theta, phi, c0, time_offset,
            transmit_delays_s_arr, transmit_element_mask)

def list_h5_data(file: h5py.File):
    """
    Recursively lists all data in an HDF5 file.
    
    Args:
        file: An open h5py.File object.

    Prints:
        - Dataset path.
        - Value and data type if scalar.
        - Shape, dimensions, and data type if array.
    """
    def traverse_group(group, path=""):
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
                traverse_group(item, item_path)
    
    traverse_group(file)
    # Example usage:
    # with h5py.File("example_file.h5", "r") as file:
    #     list_h5_data(file)


def find_envelope_peak_3d(data):
    """
    Find the peak of the envelope and its index along the third dimension using Hilbert transform.
    
    Parameters:
    data : ndarray
        Input array of shape (n1, n2, n3)
    
    Returns:
    peak_values : ndarray
        Peak values of the envelope, shape (n1, n2)
    peak_indices : ndarray
        Indices where peaks occur, shape (n1, n2)
    """
    from scipy.signal import hilbert

    # Get array dimensions
    n1, n2, n3 = data.shape
    
    # Initialize output arrays
    peak_values = np.zeros((n1, n2))
    peak_indices = np.zeros((n1, n2), dtype=int)
    
    # Loop through first and second dimensions
    for i in range(n1):
        for j in range(n2):
            # Get the signal for this slice
            signal = data[i, j, :]
            
            # Calculate analytic signal using Hilbert transform
            analytic_signal = hilbert(signal)
            
            # Get envelope
            envelope = np.abs(analytic_signal)
            
            # Find peak of envelope
            peak_idx = np.argmax(envelope)
            peak_values[i, j] = envelope[peak_idx]
            peak_indices[i, j] = peak_idx
    
    return peak_values, peak_indices


def apply_hann_window(ap, fwhm_x=None, fwhm_y=None):
    if fwhm_x is not None:
        x_window = np.power(np.hanning(ap.shape[0]), 1/fwhm_x)
        ap = ap * x_window[:, None]
    if fwhm_y is not None:
        y_window = np.power(np.hanning(ap.shape[1]), 1/fwhm_y)
        ap = ap * y_window[None, :]
    return ap


def interpolate_3d_xyz(coords, data_3d, xaxis, yaxis, zaxis):
    """
    Interpolates the 3D matrix (data_3d) on the given list of coordinates.
    Here, we assume that:
      - The data_3d array has the shape (len(xaxis), len(yaxis), len(zaxis))
        (i.e., indexing="ij" style)
      - coords is an (N x 3) array, where each coordinate is [x, y, z].

    This is consistent with a call like:

        np.meshgrid(xaxis, yaxis, zaxis, indexing="ij")

    followed by reshaping into (N, 3).

    Parameters:
    -----------
    coords : np.ndarray
        A (N x 3) array of 3D coordinates of the form [x, y, z].
    data_3d : np.ndarray
        A 3D array representing data over a grid defined by (xaxis, yaxis, zaxis).
        Must have shape (len(xaxis), len(yaxis), len(zaxis)).
    xaxis : np.ndarray
        1D array representing the x-coordinates of the 3D grid.
    yaxis : np.ndarray
        1D array representing the y-coordinates of the 3D grid.
    zaxis : np.ndarray
        1D array representing the z-coordinates of the 3D grid.

    Returns:
    --------
    np.ndarray
        A 1D array of interpolated values corresponding to each coordinate in coords.
    """
    # Ensure the inputs are numpy arrays
    coords = np.asarray(coords, dtype=np.float32)
    xaxis = np.asarray(xaxis, dtype=np.float32)
    yaxis = np.asarray(yaxis, dtype=np.float32)
    zaxis = np.asarray(zaxis, dtype=np.float32)

    # Check if provided axis lengths match data dimensions.
    nx, ny, nz = data_3d.shape
    if len(xaxis) != nx:
        print(f"Warning: xaxis length {len(xaxis)} does not match data_3d.shape[0] ({nx}). Reconstructing xaxis.")
        xaxis = np.linspace(xaxis[0], xaxis[-1], nx, dtype=np.float32)
    if len(yaxis) != ny:
        print(f"Warning: yaxis length {len(yaxis)} does not match data_3d.shape[1] ({ny}). Reconstructing yaxis.")
        yaxis = np.linspace(yaxis[0], yaxis[-1], ny, dtype=np.float32)
    if len(zaxis) != nz:
        print(f"Warning: zaxis length {len(zaxis)} does not match data_3d.shape[2] ({nz}). Reconstructing zaxis.")
        zaxis = np.linspace(zaxis[0], zaxis[-1], nz, dtype=np.float32)

    # Create the interpolator
    # Note that for RegularGridInterpolator, the data is assumed to be ordered as:
    # data_3d[i, j, k] = value at (xaxis[i], yaxis[j], zaxis[k])
    interpolator = RegularGridInterpolator((xaxis, yaxis, zaxis), data_3d)

    # Interpolate the values at the given coordinates
    interpolated_values = interpolator(coords)

    return interpolated_values
def compute_arrive_time(
    lats: np.ndarray,
    els: np.ndarray,
    deps: np.ndarray,
    idt0: float,
    theta: float,
    phi: float,
    dt: float,
    c0: float
) -> np.ndarray:
    """
    Calculate the arrival time matrix.

    Parameters:
    - lats (np.ndarray): Array of latitudes with shape (nlat,).
    - els (np.ndarray): Array of elevations with shape (nel,).
    - deps (np.ndarray): Array of depths with shape (ndep,).
    - idt0 (float): Base arrival time.
    - theta (float): Angle theta in radians.
    - phi (float): Angle phi in radians.
    - dt (float): Time step.
    - c0 (float): Wave speed constant.

    Returns:
    - idt_matrix (np.ndarray): Flattened array of calculated arrival times with shape (nlat * nel * ndep,).
    """
    return idt0 + (
        lats[:, None, None] * np.sin(theta) * np.cos(phi) +
        els[None, :, None] * np.sin(theta) * np.sin(phi) +
        deps[None, None] * np.cos(theta)
    ) / (dt * c0)
def remove_mean_delays(delays):
    """Subtract the nanmean from each slice (ignoring NaNs)."""
    delays_zm = np.empty_like(delays)
    mean_delays = np.empty(delays.shape[0], dtype=delays.dtype)
    for i in range(delays.shape[0]):
        mean_val = np.nanmean(delays[i])
        mean_delays[i] = mean_val
        delays_zm[i] = delays[i] - mean_val
    return delays_zm, mean_delays

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

def create_aperture_mask(nX, nY, xaxis, yaxis, wX, wY, fwhm_x=0, fwhm_y=0):
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
            If set to 0, no windowing is applied in that dimension.
            
    Returns:
        np.ndarray
            The (optionally windowed) aperture mask.
    """
    import numpy as np  # In case it's not already imported.
    
    X, Y = np.meshgrid(xaxis, yaxis, indexing='ij')
    # Create a binary aperture: ones inside the aperture, zeros outside.
    ap = np.where((np.abs(X) <= wX/2) & (np.abs(Y) <= wY/2), 1.0, 0.0)
    
    # Determine whether to apply the Hann window. If a fwhm parameter is zero,
    # pass None so that no window is applied in that dimension.
    fwhm_x_arg = None if fwhm_x == 0 else fwhm_x
    fwhm_y_arg = None if fwhm_y == 0 else fwhm_y
    if fwhm_x_arg is not None or fwhm_y_arg is not None:
        ap = apply_hann_window(ap, fwhm_x=fwhm_x_arg, fwhm_y=fwhm_y_arg)
    
    # Ensure that values outside the aperture remain zero.
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

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
def plot_propagation_step(apaz, pI, pnp, pax, arrivaltime_matrix, amplitude_mask,
                          xaxis, yaxis, t, zaxis, current_dist, cc,
                          dT, c0, pdur, f0, rho0, save_path):
    """
    Create a multi-panel plot for a propagation step and save the figure.
    
    The top 4 plots are arranged in a 2×2 grid:
      0. x-t pressure field
      1. y-t pressure field
      2. Time trace at the center
      3. 2D intensity map in the x-y plane (final z slice)
      
    The bottom 8 plots are arranged in a 2×4 grid and use the full z-axis defined
    by 'zaxis' (e.g. zaxis = np.arange(n_steps)*dZ). The vertical axis is fixed and
    flipped (z = 0 at the top), and the axes are scaled equally (physically correct).
    """
    plt.clf()
    fig = plt.figure(figsize=(24, 30))
    
    # Create two gridspecs: one for the top 4 subplots and one for the bottom 8.
    gs_top = gridspec.GridSpec(2, 2, top=0.95, bottom=0.6, left=0.05, right=0.95,
                               hspace=0.3, wspace=0.3)
    gs_bottom = gridspec.GridSpec(2, 4, top=0.55, bottom=0.05, left=0.05, right=0.95,
                                  hspace=0.4, wspace=0.3)
    
    # For the bottom eight plots, assume zaxis[0] == 0 (top) and zaxis[-1] is the maximum propagation distance.
    z_top = zaxis[0]      # expected to be 0
    z_bottom = zaxis[-1]  # maximum propagation distance

    # === Top 4 Plots (2x2 grid) ===
    # Plot 0: x-t pressure field.
    ax = fig.add_subplot(gs_top[0, 0])
    im = ax.imshow(apaz[:, len(yaxis)//2, :].T,
                   extent=[xaxis[0], xaxis[-1], t[0], t[-1]],
                   aspect='auto')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('t (s)')
    ax.set_title(f'Pressure field at z = {current_dist:.3f} m')
    plt.colorbar(im, ax=ax, label='Pa')
    
    # Plot 1: y-t pressure field.
    ax = fig.add_subplot(gs_top[0, 1])
    im = ax.imshow(apaz[len(xaxis)//2, :, :].T,
                   extent=[yaxis[0], yaxis[-1], t[0], t[-1]],
                   aspect='auto')
    ax.set_xlabel('y (m)')
    ax.set_ylabel('t (s)')
    ax.set_title(f'Pressure field at z = {current_dist:.3f} m')
    plt.colorbar(im, ax=ax, label='Pa')
    
    # Plot 2: Time trace at center.
    ax = fig.add_subplot(gs_top[1, 0])
    ax.plot(t, apaz[len(xaxis)//2, len(yaxis)//2, :])
    ax.set_xlabel('t (s)')
    ax.set_ylabel('Pa')
    ax.set_title(f'Center pressure at z = {current_dist:.3f} m')
    ax.grid(True)
    
    # Plot 3: 2D intensity map in x-y plane (using the final z slice).
    ax = fig.add_subplot(gs_top[1, 1])
    isppa = pI[:, :, cc] * dT / (c0 * rho0 * pdur) / 10000
    im = ax.imshow(isppa.T,
                   extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
                   aspect='equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Isppa (W/cm²)')
    plt.colorbar(im, ax=ax, label='W/cm²')
    
    # === Bottom 8 Plots (2x4 grid) ===
    # Loop over 8 plots (indices 4 to 11) with 2 rows and 4 columns.
    for j in range(8):
        i = j + 4  # overall subplot index (from 4 to 11)
        row = j // 4  # 0 or 1 (2 rows)
        col = j % 4   # 0,1,2,3 (4 columns)
        ax = fig.add_subplot(gs_bottom[row, col])
        if i in [4, 5]:
            # Intensity profiles along a fixed cross-section vs. z.
            if i == 4:
                data = pI[:, len(yaxis)//2, :] * dT / (c0 * rho0 * pdur) / 10000
                x_or_y, xlabel = xaxis, 'x (m)'
            else:
                data = pI[len(xaxis)//2, :, :] * dT / (c0 * rho0 * pdur) / 10000
                x_or_y, xlabel = yaxis, 'y (m)'
            im = ax.imshow(data.T,
                           extent=[x_or_y[0], x_or_y[-1], z_bottom, z_top],
                           aspect='equal')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('z (m)')
            ax.set_title('Isppa (W/cm²)')
            ax.set_ylim(z_bottom, z_top)
            plt.colorbar(im, ax=ax, label='W/cm²')
        elif i in [6, 7]:
            # MI profiles over the full z-axis.
            if i == 6:
                mi_data = -pnp[:, len(yaxis)//2, :] / 1e6 / np.sqrt(f0/1e6)
                x_or_y, xlabel = xaxis, 'x (m)'
            else:
                mi_data = -pnp[len(xaxis)//2, :, :] / 1e6 / np.sqrt(f0/1e6)
                x_or_y, xlabel = yaxis, 'y (m)'
            im = ax.imshow(mi_data.T,
                           extent=[x_or_y[0], x_or_y[-1], z_bottom, z_top],
                           aspect='equal')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('z (m)')
            ax.set_title('MI')
            ax.set_ylim(z_bottom, z_top)
            plt.colorbar(im, ax=ax, label='MI')
        elif i in [8, 9]:
            # Arrival time profiles over the full z-axis.
            if i == 8:
                data = arrivaltime_matrix[:, len(yaxis)//2, :]
                x_or_y, xlabel = xaxis, 'x (m)'
            else:
                data = arrivaltime_matrix[len(xaxis)//2, :, :]
                x_or_y, xlabel = yaxis, 'y (m)'
            im = ax.imshow(data.T,
                           extent=[x_or_y[0], x_or_y[-1], z_bottom, z_top],
                           aspect='equal')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('z (m)')
            ax.set_title('Arrival Times')
            ax.set_ylim(z_bottom, z_top)
            plt.colorbar(im, ax=ax, label='s')
        elif i in [10, 11]:
            # Amplitude mask profiles over the full z-axis.
            if i == 10:
                data = amplitude_mask[:, len(yaxis)//2, :]
                x_or_y, xlabel = xaxis, 'x (m)'
            else:
                data = amplitude_mask[len(xaxis)//2, :, :]
                x_or_y, xlabel = yaxis, 'y (m)'
            im = ax.imshow(data.T,
                           extent=[x_or_y[0], x_or_y[-1], z_bottom, z_top],
                           aspect='equal')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('z (m)')
            ax.set_title('Amplitude Mask')
            ax.set_ylim(z_bottom, z_top)
            plt.colorbar(im, ax=ax, label='Amplitude')
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()  # Close the figure so it is not displayed
######################################################################
#######################################################################
print(jax.devices())


file_name = '/home/gfp/Downloads/seq-0-IQ_3D_1cmto6cm_centered_Depth_5MHz_3a/schema/ensemble_2024-12-12T00-20-34-651724.h5'
#file_name = '/Users/gianmarcopinton/seq-0-IQ_3D_1cmto6cm_centered_Depth_5MHz_3a/schema/ensemble_2024-12-12T00-20-34-651724.h5'
file_name = '/celerina/gfp/mfs/mangrove-test-data/phantom/ATS539_brights_rf.h5'
#file_name = '/celerina/gfp/mfs/mangrove-test-data/sub-UCLA006/ultrasound/sub-UCLA006_ses-20240607_task-audio_acq-2d_run-06_rf.h5'
file_name = '/celerina/gfp/mfs/20240909_3D_focus_beam/run-01/acquisitions/seq-RF_FullTX_FullRX_6500000Hz_1a_3c_5mmfoc_200loops_0Gain_3AFE_0rp/schema/ensemble_2024-09-09T23-40-27-496903.h5'


with h5py.File(file_name, "r") as file:
    list_h5_data(file)

with h5py.File(file_name, 'r') as file:
    rf, fs, f0, lats, els, deps, theta, phi, c0, time_offset, transmit_delays_s, transmit_element_mask = \
        _get_raw_metadata(file)

idt0 = -fs * time_offset

# Remove mean from transmit delays and print range info.
transmit_delays_s_zm, mean_delays = remove_mean_delays(transmit_delays_s-time_offset)
# Remove time zero from transmit delays and print range info.
transmit_delays_s_zm = transmit_delays_s - np.nanmax(transmit_delays_s) 

# --- Physical and Domain Parameters ---
#f0    = 3e6         # Frequency (Hz)
#c0    = 1500        # Speed of sound (m/s)
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


for ff in range(transmit_delays_s.shape[0]):
    
    
    # Construct output directories.
    basedir = os.path.join(os.path.dirname(file_name), f"asr_{ff}").replace('.', 'p').replace('-1', 'w')
    #fstr = f'/celerina/gfp/mfs/mangrove-test-data/sub-UCLA006/ultrasound/asr_{ff}'.replace('.', 'p').replace('-1', 'w')
    #basedir = os.path.join('/celerina/gfp/mfs/angularspectrum_python', fstr)
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
    
    # need to adjust duration to be greater than max and min delays
    # --- Time Axis and Initial Condition ---
    ncycles = 1.5
    dur = 2
    duration_min = ncycles * 6 * 2 * np.pi / omega0
    duration = 2.2*(np.nanmax(transmit_delays_s_zm)-np.nanmin(transmit_delays_s_zm))
    if duration < duration_min:
        duration = duration_min
    nT = round(duration / dT)
    if nT % 2 == 0:
        nT += 1
    #t = np.arange(nT) * dT
    #t = np.arange(nT) * dT - 2 * ncycles / omega0 * 2 * np.pi
    t = np.arange(nT) * dT - duration/2
    t -= np.mean(t)
    icvec = create_initial_condition(t, omega0, ncycles, dur, p0)
    plt.figure()
    plt.plot(t, icvec)
    plt.grid(True)
    plt.title('Initial condition (Pa)')
    plt.xlabel('t (s)'); plt.ylabel('Pa')
    plt.savefig(os.path.join(basedir, 'figures', 'icvec.png'), dpi=400)
    plt.close()

    #icvec_pI = np.sum(icvec**2)
    #icvec_t0 = t[peak_idxs]

    pdur = np.sum(np.abs(signal.hilbert(icvec)) > p0/2) * dT
    intensity = np.sum(icvec**2) * dT / (c0 * rho0 * pdur)
    intensity_per_cm2 = intensity / 10000

    # --- Aperture Mask ---
    ap = create_aperture_mask(nX, nY, xaxis, yaxis, wX, wY,0,0)
    plt.figure()
    plt.imshow(ap.T, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]], aspect='equal')
    plt.title('Aperture mask')
    plt.xlabel('x (m)'); plt.ylabel('y (m)')
    plt.savefig(os.path.join(basedir, 'figures', 'ic_mask.png'), dpi=400)
    plt.close()

    # --- Space-Time Field ---
    # Use the first slice of interpolated delays as the delay map.
    tt = delays_interp[ff]
    tt[np.isnan(tt)] = 0
    #tt = tt - np.mean(tt[tt != 0])
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

    # Precalculate components using our previously translated functions
    # Note: Converting to float32 for compatibility with many GPUs
    HH = precalculate_mas(nX, nY, nT, dX, dY, dZ, dT, c0).astype(np.complex64)
    abl = precalculate_abl(nX, nY, nT, 2e-3/nY/dY).astype(np.float32)


    plt.figure()
    # In Python, we need to explicitly transpose the data for proper visualization
    # The squeeze function removes singleton dimensions, similar to MATLAB's squeeze
    plt.imshow(abl[:, nY//2, :].T, aspect='auto')
    plt.colorbar(label='Absorption Coefficient')
    plt.title('Absorbing Boundary Layer Profile')
    plt.xlabel('X Position')
    plt.ylabel('Time')
    plt.savefig(os.path.join(basedir, 'figures', 'abl.png'), dpi=400)

    # Precalculate absorption/dispersion filter based on material properties
    if a0 == -1:  # Water
        alpha0 = 2.17e-3  # dB/MHz^2/cm for water
        afilt3d = precalculate_ad(alpha0, nX, nY, nT, dZ, dT)
        pow = 2
    else:
        alpha0 = a0
        pow = 1  # dB/MHz^pow/cm power law attenuation
        afilt3d, _, _, _ = precalculate_ad_pow2(alpha0, nX, nY, nT, dZ, dT, c0, omega0/(2*np.pi), pow)

 
    # Convert operators 
    HH = jnp.array(HH, dtype=jnp.complex64)
    abl = jnp.array(abl, dtype=jnp.float32)
    afilt3d = jnp.array(afilt3d*0+1, dtype=jnp.complex64)

    # Initialize arrays to store propagation results
    n_steps = int(np.ceil(prop_dist/dZ))+1
    pnp = np.zeros((nX, nY, n_steps), dtype=np.float32)  # Peak negative pressure
    ppp = np.zeros((nX, nY, n_steps), dtype=np.float32)  # Peak positive pressure
    pI = np.zeros((nX, nY, n_steps), dtype=np.float32)   # Intensity
    pax = np.zeros((nT, n_steps), dtype=np.float32)      # Axial pressure
    arrivaltime_matrix = np.zeros((nX, nY, n_steps), dtype=np.float32)  # Arrival time matrix
    amplitude_mask = np.zeros((nX, nY, n_steps)).astype(np.float32)  # Amplitude mask
    zaxis = np.arange(n_steps) * dZ

    # Initialize propagation variables
    apaz = jnp.array(apa, dtype=jnp.float32)  # Current field
    zvec = [0]                    # Propagation distance vector
    cc = 0                        # Step counter
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

        # Check stability criterion
        if N * dZ/dT * np.max(np.abs(apaz)) > 0.1:
            print('Stability criterion violated, retrying with smaller step size')
            dZ = 0.075 * dT / (np.max(np.abs(apaz)) * N)
            n_steps = int(np.ceil(prop_dist / dZ))
            cc=0;
            zvec = [0]

            # Recalculate operators with new step size
            HH = precalculate_mas(nX, nY, nT, dX, dY, dZ, dT, c0).astype(np.complex64)
            abl = precalculate_abl(nX, nY, nT, 2e-3/nY/dY).astype(np.float32)
            if a0 == -1:
                afilt3d = precalculate_ad(alpha0, nX, nY, nT, dZ, dT)
            else:
                afilt3d, _, _, _ = precalculate_ad_pow2(alpha0, nX, nY, nT, dZ, dT, c0, 
                                                    omega0/(2*np.pi), pow)
            
            # Move to device
            HH = jnp.array(HH, dtype=jnp.complex64)
            abl = jnp.array(abl, dtype=jnp.float32)
            afilt3d = jnp.array(afilt3d*0+1, dtype=jnp.complex64)
            continue
        
        # Current propagation distance
        current_dist = sum(zvec)
        print(f'Propagation distance = {current_dist:.6f} m')
        
        # Perform one propagation step using JAX
        apaz = march_step(apaz, HH, abl, afilt3d, N, dZ, dT)
        
        # Plot and save the propagation state.
        plot_path = os.path.join(fig_dir, f'agp_{cc:04d}.jpg')
        plot_propagation_step(apaz, pI, pnp, pax, arrivaltime_matrix, amplitude_mask,
                                xaxis, yaxis, t, zaxis, current_dist, cc,
                                dT, c0, pdur, f0, rho0, plot_path)
        zvec.append(dZ)
        cc += 1
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Simulation run time: {elapsed_time:.2f} seconds")

    
    z=np.cumsum(zvec)

    # Save final results
    np.savez(os.path.join(basedir, 'results.npz'),
            pnp=pnp, ppp=ppp,
            pI=pI, pax=pax,
            x=xaxis, y=yaxis, t=t, z=z)


    print("Simulation completed successfully.")

    # define the grid
    bmode_lats = lats  # lats[20:-20]
    bmode_els = els  # bmode_lats
    bmode_deps = deps  # deps[200:300]

    beamforming_grid = np.ascontiguousarray(
        np.array(
            np.meshgrid(bmode_lats, bmode_els, bmode_deps, indexing="ij")
        )
        .reshape(3, -1)
        .T
    ).astype(np.float32)
    
    peak_idx = np.argmax(icvec) # calculate residual offset from icvec
    idt00 = taxis[peak_idx]

    arrivaltime_matrix_interpolated = interpolate_3d_xyz(beamforming_grid, arrivaltime_matrix, xaxis, yaxis, z)
    arrivaltime_matrix_interpolated = arrivaltime_matrix_interpolated.reshape(len(bmode_lats), len(bmode_els), len(bmode_deps)) 
    #arrivaltime_matrix_interpolated = arrivaltime_matrix_interpolated+idt0

    # compare to analytical function
    idt0 = -fs * time_offset
    idt_matrix = compute_arrive_time(lats, els, deps,
                                     idt0, theta, phi, 1 / fs, c0)
    #idt_matrix=idt_matrix.flatten()
    j=20 #for i in range(arrivaltime_matrix_interpolated.shape[2]):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(arrivaltime_matrix_interpolated[:, j, :].T*fs+idt0, aspect='auto')
    plt.colorbar(label='Arrival Time')
    plt.title(f'Simulated Arrival Time - Slice {j + 1}')
    plt.xlabel('X')
    plt.ylabel('Z')

    plt.subplot(1, 2, 2)
    plt.imshow(idt_matrix[:, j, :].T, aspect='auto')
    plt.colorbar(label='Arrival Time')
    plt.title(f'Analytical Arrival Time - Slice {j + 1}')
    plt.xlabel('X')
    plt.ylabel('Z')

    plt.tight_layout()
    #plt.show(block=False)
    plt.savefig('at_xz.png', dpi=400)

    #plt.pause(1)
    #plt.close
    np.nanmin(arrivaltime_matrix_interpolated*fs+idt0-idt00*fs)
    np.nanmax(arrivaltime_matrix_interpolated*fs+idt0-idt00*fs)
    np.nanmin(idt_matrix)
    np.nanmax(idt_matrix)

    np.nanmax(arrivaltime_matrix)
    np.nanmin(arrivaltime_matrix)
    np.savez(os.path.join(basedir, 'at_variables.npz'),
             arrivaltime_matrix=arrivaltime_matrix-idt00,
             pI=pI,
             xaxis=xaxis,
             yaxis=yaxis,
             z=z,)


  