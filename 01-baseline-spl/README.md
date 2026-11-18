# Gravothermal Evolution with a Single-Power-Law (SPL) Baryon Profile

This directory contains the numerical pipeline used to reproduce and benchmark the gravothermal core-collapse evolution of a Self-Interacting Dark Matter (SIDM) halo embedded in a static **Single-Power-Law (SPL)** baryonic potential, following the setup of **Feng et al. (2025)**.

## Attribution & Modifications
- **Upstream Code:** Adapted from the 1D gravothermal fluid solver [`ymzhong/gravothermal`](https://github.com/ymzhong/gravothermal) by Yi-Ming Zhong.
- **Key Modifications:** 
  - Replaced the default Plummer/Hernquist static baryonic potentials with a **Single-Power-Law (SPL)** enclosed mass profile M<sub>b</sub>(r) ∝ r<sup>0.6</sup> (`MbaryonSPL` in `evolution.cpp` and `mass_baryon` / `density_baryon` in `initialization.py`).
  - Updated the Jeans equation solver and initial luminosity profile (`vd_dm`, `big_dev`, `luminosity_dm`) in `initialization.py` to self-consistently account for the SPL baryonic gravitational potential at t = 0.

---

## Pipeline Overview

### 1. Initial Conditions (`initialization.py`)
Generates the initial 1D Lagrangian grid (r<sub>min</sub> = 0.005, r<sub>max</sub> = 500, N = 180 shells) using arbitrary-precision arithmetic (`mpmath`, 25 decimal places):
- **Dark Matter Halo:** Standard NFW profile (ρ<sub>DM</sub>(r) = [r(1 + r)<sup>2</sup>]<sup>−1</sup>).
- **Baryonic Component:** Static SPL profile (M<sub>b</sub>(r) = M<sub>norm</sub> r<sup>0.6</sup>, ρ<sub>b</sub>(r) = 0.6 M<sub>norm</sub> r<sup>−2.4</sup>).
- **Outputs:** Exports radial grids, enclosed DM mass, density, specific internal energy u(r), luminosity L(r), and a diagnostic plot (`profile.pdf`) to `initial/<tag>/`.

### 2. Gravothermal Time Evolution (`evolution.cpp`)
Integrates the coupled gravothermal fluid equations (heat conduction + hydrostatic relaxation) in C++ using the **Eigen** linear algebra library (`Eigen::LLT` solver):
- **Heat Conduction:** Computes conductive energy transport across Lagrangian shells.
- **Hydrostatic Relaxation:** Enforces hydrostatic equilibrium at each timestep including the static SPL baryonic mass (`MhyList = MList + MbaryonSPL`), dynamically increasing output frequency to every step once the central density reaches ρ(0) ≥ 10<sup>6</sup>.
- **Outputs:** Exports simulation metadata and the time-evolving radial profiles of radius r, DM density ρ(r), enclosed DM mass M(r), specific internal energy u(r), and luminosity L(r) across saved timesteps to `results/result_<tag>.txt`.

---

## How to Run

### Prerequisites
- **Python 3:** `numpy`, `matplotlib`, `mpmath`
- **C++ Compiler:** `g++` (C++17 or later) with the **Eigen3** library installed

### Step 1: Generate Initial Profiles
Run the Python initialization script to generate the `initial/<tag>/` directory and initial conditions:
```bash
python initialization.py
```

### Step 2: Compile and Run the C++ Evolution Solver
Create the `results/` output folder (if not already present), compile `evolution.cpp` with `-O3` optimization pointing to your Eigen3 include path, and launch the executable:
```powershell
mkdir -Force results
g++ -O3 -std=c++17 -I C:\path\to\eigen3 evolution.cpp -o evolution.exe
.\evolution.exe
```
