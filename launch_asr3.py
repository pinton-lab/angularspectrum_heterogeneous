import jax.numpy as jnp
from jax import device_put
from jax.numpy.fft import fftn, ifftn, fftshift, ifftshift, fft, ifft
import jax
import numpy as np
from scipy import signal
import os
import matplotlib.pyplot as plt
from jax import jit, vmap

from pathlib import Path
from scipy.signal import hilbert
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

print(jax.devices())


# Basic physical parameters
f0 = 3e6  # Frequency in Hz
c0 = 1500  # Speed of sound in m/s
rho0 = 1000  # Density in kg/m^3

# Domain parameters
array_dx = 2.0800e-04  # Grid spacing
wX = 138 * array_dx  # Width in x-direction
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
p0 = 0.3e6
a0 = 0.5  # -1 indicates water
fcen = np.array([0, 0, 5]) * 1e-2

# Note: MATLAB's gpuDevice(2) command isn't needed in Python
# If GPU support is needed, you'd use a framework like CuPy or PyTorch

# Main calculation loop
fcens = np.array([[0, 0, 5e-2]])  # Example value, adjust as needed
for ff in range(fcens.shape[0]):
    fcen = fcens[ff, :]
    
    # Create directory strings
    fstr = f'asr3_butterfly4_beamanalysis_receive_multi3d_f3_gpu2_foc_simple{ff}'
    fstr = fstr.replace('.', 'p').replace('-1', 'w')
    basedir = os.path.join('/celerina/gfp/mfs/angularspectrum_python', fstr)
    
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
    bdy = 1e-2

    # Calculate grid dimensions
    nX = 2 * round((wX + bdy) / dX / 2) + 1
    nY = 2 * round((wY + bdy) / dX / 2) + 1
    
    # Create coordinate axes
    xaxis = np.arange(nX) * dX - (wX/2 + bdy/2)
    yaxis = np.arange(nY) * dX - (wY/2 + bdy/2)

    # Generate initial conditions
    ncycles = 6
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
            if abs(xaxis[i]) <= wX/2 and abs(yaxis[j]) <= wYsub/2:
                ap[i, j] = 1

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
    tt = tt * ap
    tt = tt - np.max(tt[tt != 0])
    tt = tt + duration/2 - ncycles * 2 * np.pi / omega0 * 2
    tt = tt * ap

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
    n_steps = int(np.ceil(prop_dist/dZ))
    pnp = np.zeros((nX, nY, n_steps), dtype=np.float32)  # Peak negative pressure
    ppp = np.zeros((nX, nY, n_steps), dtype=np.float32)  # Peak positive pressure
    pI = np.zeros((nX, nY, n_steps), dtype=np.float32)   # Intensity
    pax = np.zeros((nT, n_steps), dtype=np.float32)      # Axial pressure

    # Initialize propagation variables
    apaz = jnp.array(apa, dtype=jnp.float32)  # Current field
    zvec = [dZ]                    # Propagation distance vector
    cc = 0                        # Step counter

    # Create a figure for real-time visualization
    fig = plt.figure(figsize=(20, 24))

    # Main propagation loop
    while sum(zvec) < prop_dist:
        # Store axial pressure (center of domain)
        pax[:, cc] = apaz[nX//2, nY//2, :]
        
        # Calculate intensity
        pI[:, :, cc] = np.sum(apaz**2, axis=2)
        
        # Store peak negative and positive pressures
        pnp[:, :, cc] = np.min(apaz, axis=2)
        ppp[:, :, cc] = np.max(apaz, axis=2)
        
        # Check stability criterion
        if N * dZ/dT * np.max(np.abs(apaz)) > 0.1:
            print('Stability criterion violated, retrying with smaller step size')
            dZ = 0.075 * dT / (np.max(np.abs(apaz)) * N)
            
            # Recalculate operators with new step size
            HH = precalculate_mas(nX, nY, nT, dX, dY, dZ, dT, c0).astype(np.float32)
            abl = precalculate_abl(nX, nY, nT, 2e-3/nY/dY).astype(np.float32)
            if a0 == -1:
                afilt3d = precalculate_ad(alpha0, nX, nY, nT, dZ, dT)
            else:
                afilt3d, _, _, _ = precalculate_ad_pow2(alpha0, nX, nY, nT, dZ, dT, c0, 
                                                    omega0/(2*np.pi), pow)
            
            # Move to device
            HH = jnp.array(HH)
            abl = jnp.array(abl)
            afilt3d = jnp.array(afilt3d)
            continue
        
        # Current propagation distance
        current_dist = sum(zvec)
        print(f'Propagation distance = {current_dist:.6f} m')
        
        # Perform one propagation step using JAX
        apaz = march_step(apaz, HH, abl, afilt3d, N, dZ, dT)
        
        # Create visualization plots
        plt.clf()  # Clear the figure
        
        # Create 8 subplots (4x2 grid)
        for i in range(8):
            plt.subplot(4, 2, i+1)
            
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
        
        plt.tight_layout()
        plt.savefig(os.path.join(basedir, 'movies', f'agp_{cc:04d}.jpg'), dpi=200)
        
        zvec.append(dZ)
        cc += 1

    # Save final results
    np.savez(os.path.join(basedir, 'results.npz'),
            pnp=pnp[:,:,:cc], ppp=ppp[:,:,:cc],
            pI=pI[:,:,:cc], pax=pax[:,:cc],
            x=xaxis, y=yaxis, t=t, z=np.cumsum(zvec[:cc]))

    print("Simulation completed successfully.")