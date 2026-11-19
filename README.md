# SIDM Master Thesis 

Conducting-fluid 1D Lagrangian numerical solver (C++/Python) for simulating the gravothermal core collapse of Self-Interacting Dark Matter (SIDM) halos, both in the presence of static baryonic potentials and in the pure dark matter regime (implicit Crank-Nicolson scheme with Adaptive Mesh Refinement).

---

## Repository Structure

```text
sidm-master-thesis/
├── literature/                      # Reference papers, analytical notes, and background material
├── simulations/                     # Numerical solvers and post-processing pipeline (see simulations/README.md)
│   ├── 01-baseline/                 # Explicit solver with static baryonic profiles (Hernquist, Plummer, SPL)
│   ├── 02-nobar-implicit-amr/       # Optimized baryon-free C++17 solver (Crank-Nicolson + tridiagonal LU + AMR)
│   └── analysis/                    # Python diagnostics, unit conversions, and convergence scripts
└── README.md                        # Repository overview
```

---

## Overview of Contents

* **[`literature/`](literature/)** — Reference literature and theoretical background on gravothermal fluid modeling of SIDM halos.
* **[`simulations/`](simulations/README.md)** — Full simulation suite and analysis scripts:
  * **[`01-baseline/`](simulations/01-baseline/README.md):** Reference explicit Lagrangian simulations incorporating static baryonic potentials (Hernquist, Plummer, and Single Power-Law profiles).
  * **[`02-nobar-implicit-amr/`](simulations/02-nobar-implicit-amr/README.md):** High-performance pure-SIDM C++17 implementation without baryons featuring an implicit Crank-Nicolson conduction step, exact tridiagonal Newton-Raphson hydrostatic relaxation, and dynamic layer splitting/coarsening.
  * **[`analysis/`](simulations/analysis/):** Post-processing scripts to evaluate layer convergence, shell redistribution, dimensionless profile evolution, and physical-unit scaling.

---

## Requirements

* **C++:** `g++` with C++17 support (no external linear algebra libraries required for `02-nobar-implicit-amr`; `Eigen` required for `01-baseline`).
* **Python:** Python 3.8+ with `mpmath`, `numpy`, `scipy`, and `matplotlib`.
