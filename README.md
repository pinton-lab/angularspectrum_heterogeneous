# Angular Spectrum Method for Nonlinear Ultrasound Propagation

A three-dimensional nonlinear acoustic propagation solver based on the modified angular spectrum method (ASM), with extensions for heterogeneous tissue and immersed curved-source transducers.

## Features

- **Split-step propagation**: Strang splitting of diffraction (FFT-based angular spectrum), nonlinear distortion (Burgers equation), and frequency-dependent attenuation with Kramers-Kronig dispersion
- **Nonlinear flux schemes**: First-order Rusanov and second-order Kurganov-Tadmor (KT) with MUSCL reconstruction, SSP-RK2, and adaptive CFL sub-cycling
- **Heterogeneous propagation**: Phase-and-amplitude screens derived from CT data for transcranial ultrasound through skull bone
- **Immersed bowl sources**: Plane-by-plane source injection for spherical-bowl transducers (e.g., TIPS annular phased array)
- **Enhanced absorbing boundaries**: Wendland C2 taper, frequency-weighted damping, and Engquist-Majda super-absorbing correction (2.4x reflection reduction)
- **Intensity-loss tracking**: Spatially resolved absorbed-energy maps for radiation-force and thermal dose calculations
- **GPU acceleration**: JAX-based implementation with JIT compilation

## Requirements

- Python 3.9+
- JAX (with GPU support recommended)
- NumPy
- SciPy
- Matplotlib
- h5py (for transcranial validation)

## Quick Start

```python
from angular_spectrum_solver import SolverParams, angular_spectrum_solve

# Define parameters
params = SolverParams(
    dX=3.08e-4, dY=3.08e-4, dT=4e-8,
    c0=1540.0, rho0=1000.0,
    beta=3.5, alpha0=0.5, f0=1e6,
    propDist=0.065,
    useSplitStep=True,
    fluxScheme='kt',
    boundaryProfile='wendland',
    useFreqWeightedBoundary=True,
    useSuperAbsorbing=True,
)

# initial_field: ndarray of shape (nX, nY, nT)
field, pnp, ppp, pI, pIloss, zaxis, pax = angular_spectrum_solve(
    initial_field, params
)
```

### Bowl Transducer Source

```python
from angular_spectrum_solver import make_bowl_source_planes

source_planes, bowl_depth = make_bowl_source_planes(
    xaxis, yaxis, taxis,
    f0=1e6, c0=1540.0, p0=400e3,
    radius=46e-3, roc=80e-3, dZ=dZ,
    focus=50e-3, inner_radius=20.5e-3,
    n_elements=8,
)

# First slice is the initial field; remaining are injected during propagation
initial_field = source_planes[0][1]
params.sourcePlanes = source_planes[1:]
```

## Validation

The solver is validated through:

1. **Diffraction**: Baffled-piston and focused-piston benchmarks against Rayleigh-Fresnel and Airy analytical solutions
2. **Attenuation**: Power-law attenuation and Kramers-Kronig dispersion across 0.5-10 MHz
3. **Nonlinearity**: Riemann problem and shock-forming Burgers benchmarks
4. **Boundary treatments**: Cumulative 2.4x reflection reduction
5. **Convergence**: Strang+KT achieves second-order convergence rate
6. **Transcranial**: 1.1% RMS focal-plane agreement with FDTD through *ex vivo* human skull
7. **Bowl transducer**: 2.3% focal-depth agreement with FDTD for TIPS annular array

Run the validation scripts:

```bash
python validate_analytical.py    # Diffraction and nonlinear benchmarks
python validate_solver.py        # Convergence and flux comparison
python validate_boundaries.py   # Boundary treatment tests
python validate_transcranial.py  # Transcranial FDTD comparison
python validate_bowl.py          # Bowl transducer FDTD comparison
```

## Reference

G.F. Pinton, "An Angular Spectrum Method for Nonlinear Propagation in Heterogeneous Tissue with Immersed Sources for Transcranial and Therapeutic Ultrasound," 2026.

## License

This project is licensed under the GNU Lesser General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
