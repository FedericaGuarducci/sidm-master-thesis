# SIDM Gravothermal Evolution Simulations

1D Lagrangian fluid simulation suite for modeling the gravothermal core collapse of Self-Interacting Dark Matter (SIDM) halos. The repository contains the baseline reference implementations (with baryonic potentials), an optimized baryon-free Crank-Nicolson solver with Adaptive Mesh Refinement (AMR), and Python post-processing scripts.

---

## Repository Structure

```text
simulations/
├── 01-baseline/               # Original reference codes with baryonic profiles
│   ├── hernquist/             # SIDM + Hernquist baryonic profile
│   ├── plummer/               # SIDM + Plummer baryonic profile
│   ├── spl/                   # SIDM + Single Power-Law (SPL) baryonic profile
│   └── README.md              # Documentation for baseline simulations
├── 02-nobar-implicit-amr/     # Optimized pure-SIDM solver (Crank-Nicolson + AMR)
│   ├── evolution.cpp          # C++17 implicit solver with dynamic layer splitting/merging
│   ├── initialization.py      # Arbitrary-precision NFW initial condition generator
│   └── README.md              # Detailed changelog and technical documentation
├── analysis/                  # Post-processing and plotting scripts
│   ├── convergence_layers.py  # Layer-number independence tests (180 to 1000 layers)
│   ├── dimensionless_units.py # Radial profile evolution in dimensionless units
│   ├── layers_distribution.py # Histograms of dynamic shell redistribution over time
│   └── physical_units.py      # Profile evolution converted to physical units (kpc, M☉, Myr)
└── README.md                  # This overview file
```

---

## Modules Overview

### 1. `01-baseline/`
Contains the original explicit Lagrangian code configurations modeling SIDM halos embedded in different static baryonic potentials (**Hernquist**, **Plummer**, and **SPL**). Refer to `01-baseline/README.md` for details on the original numerical setup.

### 2. `02-nobar-implicit-amr/`
Contains the upgraded pure dark matter (baryon-free) evolution code. Key improvements include:
- **Implicit Crank-Nicolson conduction** solved in 𝒪(N) via the Thomas algorithm.
- **Exact tridiagonal LU hydrostatic relaxation** (replacing Cholesky factorization) with adaptive timestep rollback.
- **Dynamic Layering (AMR):** automatic shell splitting (`ln(Rᵢ / Rᵢ₋₁) > 0.15`) and coarsening to resolve the inner core while keeping output files compact.

See `02-nobar-implicit-amr/README.md` for the full list of physics, efficiency, and algorithmic modifications.

### 3. `analysis/`
Python scripts (`numpy`, `matplotlib`) to inspect simulation outputs and reproduce validation diagnostics:
- **`dimensionless_units.py` & `physical_units.py`:** Plot density, velocity dispersion, enclosed mass, and Knudsen number profiles across snapshots.
- **`layers_distribution.py`:** Tracks how AMR redistributes Lagrangian shells across logarithmic radial bins during core collapse.
- **`convergence_layers.py`:** Compares collapse timescales and core masses across different initial grid resolutions.

---

## Quick Start Workflow

1. **Generate Initial Conditions:**
   ```bash
   cd 02-nobar-implicit-amr
   python initialization.py
   ```

2. **Compile and Run the Simulation:**
   ```bash
   g++ -O3 -std=c++17 -march=native -fno-math-errno -DNDEBUG evolution.cpp -o evolve
   ./evolve
   ```

3. **Run Analysis & Diagnostics** (the solver writes `results/result_<tag>.txt`; pass that file to the scripts, see `analysis/README.md`):
   ```bash
   cd ../analysis
   python layers_distribution.py ../02-nobar-implicit-amr/results/result_20261006_180Layers_nobar.txt
   python physical_units.py ../02-nobar-implicit-amr/results/result_20261006_180Layers_nobar.txt
   python convergence_layers.py ../02-nobar-implicit-amr/results   # needs runs with several layer numbers
   ```

   `dimensionless_units.py` is specific to the Hernquist baseline runs (`01-baseline/hernquist/results/`).
