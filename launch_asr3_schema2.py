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
    """
    Apply symmetric Hann window to an aperture with FWHM control over x and y dimensions.
    
    Parameters:
    -----------
    ap : ndarray
        Input aperture array of shape (nX, nY)
    fwhm_x : float or None, optional
        FWHM of the window in x dimension as fraction of aperture width (0 to 1)
        If None, no windowing is applied in x dimension
    fwhm_y : float or None, optional
        FWHM of the window in y dimension as fraction of aperture width (0 to 1)
        If None, no windowing is applied in y dimension
    
    Returns:
    --------
    ndarray
        Windowed aperture with same shape as input
    """
    nX, nY = ap.shape
    ap_windowed = ap.copy()
    
    # Create symmetric windows directly based on array dimensions
    if fwhm_x is not None:
        x_window = np.hanning(nX)
        # Scale the window based on FWHM if needed
        x_window = np.power(x_window, 1/fwhm_x)
        ap_windowed = ap_windowed * x_window[:, np.newaxis]
    
    if fwhm_y is not None:
        y_window = np.hanning(nY)
        # Scale the window based on FWHM if needed
        y_window = np.power(y_window, 1/fwhm_y)
        ap_windowed = ap_windowed * y_window[np.newaxis, :]
    
    return ap_windowed

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


######################################################################
#######################################################################
print(jax.devices())


file_name = '/home/gfp/Downloads/seq-0-IQ_3D_1cmto6cm_centered_Depth_5MHz_3a/schema/ensemble_2024-12-12T00-20-34-651724.h5'
#file_name = '/Users/gianmarcopinton/seq-0-IQ_3D_1cmto6cm_centered_Depth_5MHz_3a/schema/ensemble_2024-12-12T00-20-34-651724.h5'
file_name = '/celerina/gfp/mfs/mangrove-test-data/phantom/ATS539_brights_rf.h5'

with h5py.File(file_name, "r") as file:
    list_h5_data(file)

with h5py.File(file_name, 'r') as file:
    rf, fs, f0, lats, els, deps, theta, phi, c0, time_offset, transmit_delays_s, transmit_element_mask = \
        _get_raw_metadata(file)


# remove the mean from the transmit delays
transmit_delays_s_zeromean = np.zeros_like(transmit_delays_s)
for i in range(transmit_delays_s.shape[0]):
    # Use nanmean to ignore NaN values when computing the mean
    transmit_delays_s_zeromean[i,:,:] = transmit_delays_s[i,:,:] - np.nanmean(transmit_delays_s[i,:,:])

# Check ranges for original data
for i in range(transmit_delays_s.shape[0]):
    print(f"Slice {i + 1} original - min: {transmit_delays_s[i].min():.2e}, max: {transmit_delays_s[i].max():.2e}")

# Check ranges for zero-mean data
for i in range(transmit_delays_s_zeromean.shape[0]):
    print(f"Slice {i + 1} zero-mean - min: {transmit_delays_s_zeromean[i].min():.2e}, max: {transmit_delays_s_zeromean[i].max():.2e}")

# Create three plots for each slice of the first dimension
#for i in range(transmit_delays_s_zeromean.shape[0]):
#    plt.figure()
#    plt.imshow(transmit_delays_s_zeromean[i], aspect='auto', interpolation='nearest', cmap='viridis')
#    plt.colorbar(label='Transmit Delay (s)')
#    plt.title(f"Transmit Delays - Slice {i + 1}")
#    plt.xlabel("Element Index (Dimension 2)")
#    plt.ylabel("Element Index (Dimension 1)")
#    plt.tight_layout()
#    plt.show()

# pdb.set_trace() 
# interpolate transmit_delays_s_zeromean to the same size as the data


# Basic physical parameters
f0 = 3e6  # Frequency in Hz
c0 = 1500  # Speed of sound in m/s
rho0 = 1000  # Density in kg/m^3

# Domain parameters
array_dx = 2.0800e-04  # Grid spacing
wX = 140 * array_dx  # Width in x-direction
wY = 64 * array_dx   # Width in y-direction
prop_dist = 8e-2     # Propagation distance

# Material properties
beta = 3.5  # Nonlinear coefficient
N = beta / (2 * c0**3 * rho0)
omega0 = f0 * 2 * np.pi

# Array parameters
wYsubs = np.array([8, 16, 32, 64]) * array_dx
wYsub = 64 * array_dx
focs = np.arange(1, 8) * 1e-2  # Changed MATLAB 1:7 to Python equivalent
focs = 1e-2
p0 = 0.03e6
a0 = 0.5  # -1 indicates water
fcen = np.array([0, 0, 5]) * 1e-2

# Note: MATLAB's gpuDevice(2) command isn't needed in Python
# If GPU support is needed, you'd use a framework like CuPy or PyTorch

# Main calculation loop
fcens = np.array([[0, 0, 5e-2]])  # Example value, adjust as needed
for ff in range(fcens.shape[0]):
    fcen = fcens[ff, :]
    
    # Create directory strings
    fstr = f'/celerina/gfp/mfs/forest/asr3_schema{ff}'
    fstr = fstr.replace('.', 'p').replace('-1', 'w')
    basedir = os.path.join('/celerina/gfp/mfs/angularspectrum_python', fstr)
    #fstr = f'asr3_schema{ff}'
    #fstr = fstr.replace('.', 'p').replace('-1', 'w')
    #basedir = os.path.join('/Users/gianmarcopinton/', fstr)


    # Create directories
    os.makedirs(os.path.join(basedir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(basedir, 'movies'), exist_ok=True)

    # Calculate wavelength and grid parameters
    lambda_ = c0 / f0
    k = omega0 / c0
    dX = lambda_ / 5
    dY = dX
    dZ = dX * 8
    dT = dX / (5 * c0)
    bdy = 2e-2

    # Calculate grid dimensions
    nX = 2 * round((wX + bdy) / dX / 2) + 1
    nY = 2 * round((wY + bdy) / dX / 2) + 1
    
    # Create coordinate axes
    xaxis = np.arange(nX) * dX - (wX/2 + bdy/2)
    yaxis = np.arange(nY) * dX - (wY/2 + bdy/2)

    # Original grid dimensions from transmit_delays_s_zeromean
    orig_nx = transmit_delays_s_zeromean.shape[1]  # 140
    orig_ny = transmit_delays_s_zeromean.shape[2]  # 64

    # Create original coordinate grids (unpadded)
    orig_x = np.linspace(-wX/2, wX/2, orig_nx)
    orig_y = np.linspace(-wY/2, wY/2, orig_ny)

    # Create new coordinate grids (padded)
    new_x = xaxis  # Already defined in your code
    new_y = yaxis  # Already defined in your code

    # Initialize array for interpolated delays
    delays_interpolated = np.zeros((transmit_delays_s_zeromean.shape[0], len(new_x), len(new_y)))

    # Interpolate each slice
    for i in range(transmit_delays_s_zeromean.shape[0]):
        # Create interpolation function, handling NaNs
        valid_mask = ~np.isnan(transmit_delays_s_zeromean[i])
        if np.any(valid_mask):
            # Create 2D grid for interpolation
            orig_X, orig_Y = np.meshgrid(orig_x, orig_y, indexing='ij')
            
            # Get valid points and their values
            valid_points = np.column_stack((orig_X[valid_mask], orig_Y[valid_mask]))
            valid_values = transmit_delays_s_zeromean[i][valid_mask]
            
            # Create interpolator
            interpolator = interpolate.LinearNDInterpolator(valid_points, valid_values, fill_value=np.nan)
            
            # Create new coordinate grid for evaluation
            new_X, new_Y = np.meshgrid(new_x, new_y, indexing='ij')
            
            # Perform interpolation
            delays_interpolated[i] = interpolator(new_X, new_Y)
            
            # Fill remaining NaNs with nearest neighbor interpolation
            if np.any(np.isnan(delays_interpolated[i])):
                nn_interpolator = interpolate.NearestNDInterpolator(valid_points, valid_values)
                nan_mask = np.isnan(delays_interpolated[i])
                delays_interpolated[i][nan_mask] = nn_interpolator(new_X[nan_mask], new_Y[nan_mask])
        else:
            print(f"Warning: Slice {i} contains only NaN values")
            delays_interpolated[i] = np.nan

    # Verify the interpolation
    print(f"Original shape: {transmit_delays_s_zeromean.shape}")
    print(f"Interpolated shape: {delays_interpolated.shape}")
    print(f"New grid dimensions (nX, nY): ({nX}, {nY})")
   

    # Generate initial conditions
    ncycles = 1.5
    dur = 2
    duration = ncycles * 6 * 2 * np.pi / omega0
    nT = round(duration / dT)
    if (nT % 2) == 0:
        nT += 1
    
    t = np.arange(nT) * dT - 2 * ncycles / omega0 * 2 * np.pi
    t = t - np.mean(t)
    
    # Generate initial condition vector
    icvec = np.exp(-(1.05 * t * omega0 / (ncycles * np.pi))**(2*dur)) * np.sin(t * omega0) * p0

    # Plot and save initial condition
    plt.figure()
    plt.plot(t, icvec)
    plt.grid(True)
    plt.title('initial condition (Pa)')
    plt.xlabel('t (s)')
    plt.ylabel('Pa')
    plt.savefig(os.path.join(basedir, 'figures', 'icvec.png'), dpi=400)

    # Calculate pulse duration and intensity
    pdur = len(np.abs(signal.hilbert(icvec)) > p0/2) * dT
    intensity = np.sum(icvec**2) * dT / (c0 * rho0 * pdur)
    intensity_per_cm2 = intensity / 10000

    # Generate rectangular aperture
    ap = np.zeros((nX, nY))
   
    for i in range(nX):
        for j in range(nY):
            if abs(xaxis[i]) <= wX/2 and abs(yaxis[j]) <= wY/2:
                ap[i, j] = 1
   
    #ap = apply_hann_window(ap, fwhm_x=0.5, fwhm_y=0.5)
   
    for i in range(nX):
        for j in range(nY):
            if abs(xaxis[i]) > wX/2 and abs(yaxis[j]) > wY/2:
                ap[i, j] = 0

 
    # Plot aperture mask
    plt.figure()
    plt.imshow(ap.T, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]], aspect='equal')
    plt.title('Aperture mask')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.savefig(os.path.join(basedir, 'figures', 'ic_mask.png'), dpi=400)

    # Initialize space-time field
    apa = np.zeros((nX, nY, nT), dtype=np.float32)
    
    # Calculate time delays
    X, Y = np.meshgrid(xaxis, yaxis, indexing='ij')
    tt = -np.sqrt((X - fcen[0])**2 + (Y - fcen[1])**2 + fcen[2]**2) / c0
    tt = tt - np.max(tt[tt != 0])
    tt = tt + duration/2 - ncycles * 2 * np.pi / omega0 * 2
 

    # Replace geometric focus delays with interpolated delays
    # Note: assuming we want the first slice, adjust index [0] if needed
    tt = delays_interpolated[0]  # Using first slice by default

    # Apply aperture
    # Apply aperture and handle NaNs
    valid_mask = ~np.isnan(tt)
    tt[~valid_mask] = 0  # Set NaNs to 0 
    # Adjust timing reference and add offsets
    tt = tt - np.mean(tt[tt != 0])
    #tt = tt + duration/2 - ncycles * 2 * np.pi / omega0 * 2

    # Generate space-time field
    for i in range(nX):
        for j in range(nY):
            if ap[i, j] != 0:
                icvec2 = np.exp(-(1.05 * (t - tt[i,j]) * omega0 / (ncycles * np.pi))**(2*dur)) * \
                         np.sin((t - tt[i,j]) * omega0) * p0
                apa[i, j, :] = ap[i, j] * icvec2

    # Plot results
    idx = np.where(xaxis >= 0)[0][0]
    idy = np.where(yaxis >= 0)[0][0]

    # Plot x-t view
    plt.figure()
    plt.imshow(apa[:, idy, :], extent=[t[0], t[-1], xaxis[0], xaxis[-1]], aspect='auto')
    plt.title('initial condition (Pa)')
    plt.xlabel('t (s)')
    plt.ylabel('x (m)')
    plt.colorbar(label='Pa')
    plt.savefig(os.path.join(basedir, 'figures', 'ic_x.png'), dpi=400)

    # Plot y-t view
    plt.figure()
    plt.imshow(apa[idx, :, :], extent=[t[0], t[-1], yaxis[0], yaxis[-1]], aspect='auto')
    plt.title('initial condition (Pa)')
    plt.xlabel('t (s)')
    plt.ylabel('y (m)')
    plt.colorbar(label='Pa')
    plt.savefig(os.path.join(basedir, 'figures', 'ic_y.png'), dpi=400)

    # Plot x-y view
    plt.figure()
    plt.imshow(apa[:, :, nT//2].T, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]], aspect='equal')
    plt.title('initial condition (Pa)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.colorbar(label='Pa')
    plt.savefig(os.path.join(basedir, 'figures', 'ic_xy.png'), dpi=400)

    taxis = np.arange(nT) * dT

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

    # Initialize propagation variables
    apaz = jnp.array(apa, dtype=jnp.float32)  # Current field
    zvec = [0]                    # Propagation distance vector
    cc = 0                        # Step counter

    # Create a figure for real-time visualization
    fig = plt.figure(figsize=(20, 24))

    # Main propagation loop
    start_time = time.time()
    while sum(zvec) < prop_dist:
        # initialize to zero
        #pI = np.zeros((nX, nY, round(prop_dist/dZ)))
        #pnp = np.zeros((nX, nY, round(prop_dist/dZ)))
        #ppp = np.zeros((nX, nY, round(prop_dist/dZ)))
        #arrivaltime_matrix = np.zeros((nX, nY, round(prop_dist/dZ)))
        #amplitude_mask = np.zeros((nX, nY, round(prop_dist/dZ))).astype(np.float32)
        # Store axial pressure (center of domain)
        pax[:, cc] = apaz[nX//2, nY//2, :]
        
        # Calculate intensity
        pI[:, :, cc] = np.sum(apaz**2, axis=2)
        pIslice = np.sum(apaz**2, axis=2)
        
        # Store peak negative and positive pressures
        pnp[:, :, cc] = np.min(apaz, axis=2)
        ppp[:, :, cc] = np.max(apaz, axis=2)
        peak_idxs = np.argmax(apaz, axis=2)
        #peak_vals, peak_idxs = find_envelope_peak_3d(apaz)
        arrivaltime_matrix[:, :, cc] = taxis[peak_idxs]+cc*dZ/c0
        #amplitude_mask[:, :, cc] = np.zeros((nX, nY))
        amplitude_mask[:, :, cc] = np.where(np.abs(pIslice) > 0.1 * np.mean(np.mean(pIslice)), 1, 0)

        # Check stability criterion
        if N * dZ/dT * np.max(np.abs(apaz)) > 0.1:
            print('Stability criterion violated, retrying with smaller step size')
            dZ = 0.075 * dT / (np.max(np.abs(apaz)) * N)
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
        
        # Create visualization plots
        z=np.cumsum(zvec[:cc])
        plt.clf()  # Clear the figure
        
        # Create 8 subplots (4x2 grid)
        for i in range(12):
            plt.subplot(6, 2, i+1)
            
            if i == 0:  # Pressure field in x-t plane
                plt.imshow(apaz[:, nY//2, :].T, 
                        extent=[xaxis[0], xaxis[-1], t[0], t[-1]], 
                        aspect='auto')
                plt.xlabel('x (m)')
                plt.ylabel('t (s)')
                plt.title(f'Pressure field at z = {current_dist:.3f} m')
                plt.colorbar(label='Pa')
                
            elif i == 1:  # Pressure field in y-t plane
                plt.imshow(apaz[nX//2, :, :].T,
                        extent=[yaxis[0], yaxis[-1], t[0], t[-1]],
                        aspect='auto')
                plt.xlabel('y (m)')
                plt.ylabel('t (s)')
                plt.title(f'Pressure field at z = {current_dist:.3f} m')
                plt.colorbar(label='Pa')
                
            elif i == 2:  # Time trace at center
                plt.plot(t, apaz[nX//2, nY//2, :])
                plt.xlabel('t (s)')
                plt.ylabel('Pa')
                plt.title(f'Center pressure at z = {current_dist:.3f} m')
                plt.grid(True)
                
            elif i == 3:  # Intensity in x-y plane
                isppa = pI[:, :, cc] * dT/(c0*rho0*pdur)/10000
                plt.imshow(isppa.T, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
                        aspect='equal')
                plt.xlabel('x (m)')
                plt.ylabel('y (m)')
                plt.title('Isppa (W/cm²)')
                plt.colorbar(label='W/cm²')
                
            elif i == 4 or i == 5:  # Intensity profiles
                plt.imshow(pI[:, nY//2 if i == 4 else nX//2, :cc].T * dT/(c0*rho0*pdur)/10000,
                        extent=[xaxis[0] if i == 4 else yaxis[0], 
                                xaxis[-1] if i == 4 else yaxis[-1], 
                                0, current_dist],
                        aspect='equal')
                plt.xlabel('x (m)' if i == 4 else 'y (m)')
                plt.ylabel('z (m)')
                plt.title('Isppa (W/cm²)')
                plt.colorbar(label='W/cm²')
                
            elif i == 6 or i == 7:  # MI profiles
                mi_data = -pnp[:, nY//2 if i == 6 else nX//2, :cc]/1e6/np.sqrt(f0/1e6)
                plt.imshow(mi_data.T,
                        extent=[xaxis[0] if i == 6 else yaxis[0], 
                                xaxis[-1] if i == 6 else yaxis[-1], 
                                0, current_dist],
                        aspect='equal')
                plt.xlabel('x (m)' if i == 6 else 'y (m)')
                plt.ylabel('z (m)')
                plt.title('MI')
                plt.colorbar(label='MI')

            # Add new arrival time plots
            elif i == 8 or i == 9:  # New row for arrival times
                plt.imshow(arrivaltime_matrix[:, nY//2 if i == 8 else nX//2, :cc].T,  
                        extent=[xaxis[0] if i == 8 else yaxis[0], 
                                xaxis[-1] if i == 8 else yaxis[-1],
                                0, current_dist],  # Added z-axis limits
                        aspect='auto')
                plt.xlabel('x (m)' if i == 8 else 'y (m)')
                plt.ylabel('z (m)')
                plt.title('Arrival Times')
                plt.colorbar(label='s')
            # Add new arrival time plots
            elif i == 10 or i == 11:  
                plt.imshow(amplitude_mask[:, nY//2 if i == 10 else nX//2, :cc].T,  
                        extent=[xaxis[0] if i == 10 else yaxis[0], 
                                xaxis[-1] if i == 10 else yaxis[-1],
                                0, current_dist],  # Added z-axis limits
                        aspect='auto')
                plt.xlabel('x (m)' if i == 10 else 'y (m)')
                plt.ylabel('z (m)')
                plt.title('Amplitude Mask')
                plt.colorbar(label='Amplitude')
  
    
        
        plt.tight_layout()
        plt.savefig(os.path.join(basedir, 'movies', f'agp_{cc:04d}.jpg'), dpi=200)
        
        zvec.append(dZ)
        cc += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Simulation run time: {elapsed_time:.2f} seconds")

    cc=cc
    z=np.cumsum(zvec[:cc])

    # Save final results
    np.savez(os.path.join(basedir, 'results.npz'),
            pnp=pnp[:,:,:cc], ppp=ppp[:,:,:cc],
            pI=pI[:,:,:cc], pax=pax[:,:cc],
            x=xaxis, y=yaxis, t=t, z=np.cumsum(zvec[:cc]))


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
    
    arrivaltime_matrix=arrivaltime_matrix*fs

    arrivaltime_matrix_interpolated = interpolate_3d_xyz( beamforming_grid, arrivaltime_matrix, xaxis, yaxis, z)
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
    plt.imshow(arrivaltime_matrix_interpolated[:, j, :].T, aspect='auto')
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
    plt.show()

    i=20
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(arrivaltime_matrix_interpolated[i, :, :].T, aspect='auto')
    plt.colorbar(label='Arrival Time')
    plt.title(f'Simulated Arrival Time - Slice {i + 1}')
    plt.xlabel('X')
    plt.ylabel('Z')

    plt.subplot(1, 2, 2)
    plt.imshow(idt_matrix[i, :, :].T, aspect='auto')
    plt.colorbar(label='Arrival Time')
    plt.title(f'Analytical Arrival Time - Slice {i + 1}')
    plt.xlabel('X')
    plt.ylabel('Z')

    plt.tight_layout()
    plt.show()

    fnumber = 1.0
    interp_type = 2  # 0: nearest neighbor, 1: linear interpolation, 2: quadratic interpolation
    apod_flag = 1  # apodization function enabled
    alpha = 0
    time_decim = 1
    test_iq = True
    test_rf = True

    kwargs = {
        'idps_dim0': 32,
        'idps_dim1': 16,
        'idps_dim2': 1,
        'idps_n0': 2,
        'idps_n1': 18,
        'bf_dim0': 16,
        'bf_dim1': 16,
        'bf_dim2': 4,
        'bf_n0': 4
    }



    # rf = np.tile(rf.T, 200).T
    print(f"Size of rf: {rf.shape}")
    print(f"Data type of rf: {rf.dtype}")
    n_frames, n_parameters, n_elements_y, n_elements_x, n_samples = rf.shape
    n_coords = n_elements_x * n_elements_y
    rf_data = rf[:, 0].reshape(n_frames, n_coords, n_samples)

    print(f"rf_data shape: {rf_data.shape}")  # Expected shape: (nt, ncoords, nframes)

    rf_gpu_input = np.ascontiguousarray(rf_data.astype(np.float32))

    coords = np.zeros((3, lats.size * els.size), dtype=np.float32)
    coords[:2] = np.array(np.meshgrid(lats, els)).reshape(2, -1)
    coords = np.ascontiguousarray(coords.T.astype(np.float32))

    print(f"rf_gpu_input shape: {rf_gpu_input.shape}")

    bmode_lats = lats  # lats[20:-20]
    bmode_els = els  # bmode_lats
    bmode_deps = deps  # deps[200:300]

    beamforming_grid = np.ascontiguousarray(
        np.array(
            np.meshgrid(lats, els, deps, indexing="ij")
        )
        .reshape(3, -1)
        .T
    ).astype(np.float32)

    idt_matrix = compute_arrive_time(lats, els, deps,
                                     idt0, theta, phi, 1 / fs, c0)


    np.savez(os.path.join(basedir, 'at_variables.npz'),
             arrivaltime_matrix=arrivaltime_matrix,
             xaxis=xaxis,
             yaxis=yaxis,
             z=z)