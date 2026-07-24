# Exciton-Phonon Scattering Analysis

This directory contains the standalone analysis script used to compute and visualize exciton-phonon scattering dynamics from first principles. The methodology and visualizations are designed to replicate the scattering rate analysis presented in Figure 2 of *Cohen et al., Phys. Rev. Lett. 132, 126902 (2024)*.

## Overview

The script processes raw exciton-phonon coupling tensors, eigenenergies, and phonon frequencies generated from *ab initio* calculations (e.g., Quantum ESPRESSO, BerkeleyGW, EPW). It evaluates phase-space conservation laws (Bose-Einstein thermodynamics and Gaussian-broadened energy conservation) to compute scattering rates via Fermi's Golden Rule.

The analysis is split into three main physical perspectives:
1. **Mode-Resolved Rates ($\Gamma_\nu$)**: Integrates out all momentum degrees of freedom to isolate the scattering contribution of individual phonon branches.
2. **Phonon-Momentum-Resolved Rates ($\Gamma_q$)**: Integrates out the exciton momentum and phonon mode to show the directionality and magnitude of scattering induced by specific phonon wavevectors.
3. **Exciton-Momentum-Resolved Times ($\Gamma_Q^{-1}$)**: Integrates out the phonon degrees of freedom to map the total scattering lifetime of an exciton originating at a specific point in the Brillouin zone.

## Prerequisites

- **Python 3.x**
- `numpy`
- `scipy`
- `matplotlib`
- `h5py`

## Input Data

The script expects an HDF5 database named `xctph_4x4x4.h5` in the same directory. This file must contain:
- `xctph`: The 5D coupling tensor $\mathcal{G}_{SS'\nu}(Q,q)$ (Assumed to be in Rydbergs).
- `energies`: Exciton band energies (Rydbergs).
- `frequencies`: Phonon frequencies (Rydbergs).
- `Q_plus_q_map`: Reciprocal space mapping indices for $Q+q$.
- `Qpts` & `qpts`: The explicit 3D momentum grids for excitons and phonons.
- `nQ`, `nq`: Grid dimensions.

## Usage

Run the script directly from the command line:

```bash
python rates_per_phonon_mode_extraction.py