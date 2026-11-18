# 1D Gravothermal Fluid Simulation Pipelines

This directory contains the 1D gravothermal fluid simulation pipelines used to investigate the core-collapse evolution of a Self-Interacting Dark Matter (SIDM) halo embedded in different static baryonic potentials.

The numerical framework and gravothermal fluid solver are adapted from the public repository [`ymzhong/gravothermal`](https://github.com/ymzhong/gravothermal) (accompanying Feng et al., 2025), extended here to compare multiple analytical baryonic distributions.

## Directory Structure

```text
01-baryon-profiles/
├── README.md        # Overview, physical models, and execution guide
├── spl/             # Literature benchmark: Single Power-Law (SPL) baryon profile
├── plummer/         # Thesis extension: Cored Plummer baryon sphere
└── hernquist/       # Thesis extension: Cuspy Hernquist baryon sphere
```

## Physical Models

### Dark Matter Halo (Common to All Subdirectories)

Every simulation initializes the dark matter component as a standard Navarro–Frenk–White (NFW) halo, normalized in units of the NFW scale radius *r*<sub>s</sub> and characteristic density *ρ*<sub>s</sub>:

* **Normalized DM Density:**
  *ρ*<sub>DM</sub>(*r*) = 1 / [*r*(1 + *r*)<sup>2</sup>]
* **Normalized DM Enclosed Mass:**
  *M*<sub>DM</sub>(*r*) = ln(1 + *r*) &minus; *r* / (1 + *r*)

### Static Baryonic Potentials

Each subdirectory implements a different analytical static baryonic density *ρ*<sub>b</sub>(*r*) and enclosed mass *M*<sub>b</sub>(*r*), where *r* is in units of *r*<sub>s</sub>, *a*<sub>s</sub> = *a* / *r*<sub>s</sub> is the normalized baryonic scale radius (`0.1` by default), and *M*<sub>0</sub> = *M*<sub>b</sub> / (4&pi;*ρ*<sub>s</sub>*r*<sub>s</sub><sup>3</sup>) is the normalized baryonic mass parameter (`0.1` by default):

| Subdirectory | Baryon Profile | Role in Thesis | Normalized Density *ρ*<sub>b</sub>(*r*) | Normalized Enclosed Mass *M*<sub>b</sub>(*r*) |
| :--- | :--- | :--- | :--- | :--- |
| `spl/` | Single Power-Law (SPL) | Benchmark (Feng et al., 2025) | 0.07 &middot; *r*<sup>&minus;2.4</sup> | *M*<sub>0</sub> &middot; *r*<sup>0.6</sup> |
| `plummer/` | Plummer Sphere | Exploratory Extension (Cored) | (3*M*<sub>0</sub> / *a*<sub>s</sub><sup>3</sup>) &middot; (1 + *r*<sup>2</sup>/*a*<sub>s</sub><sup>2</sup>)<sup>&minus;5/2</sup> | *M*<sub>0</sub> &middot; (1 + *a*<sub>s</sub><sup>2</sup>/*r*<sup>2</sup>)<sup>&minus;3/2</sup> |
| `hernquist/` | Hernquist Sphere | Exploratory Extension (Cuspy) | (2*M*<sub>0</sub>*a*<sub>s</sub>) / [*r*(*r* + *a*<sub>s</sub>)<sup>3</sup>] | *M*<sub>0</sub> &middot; (1 + *a*<sub>s</sub>/*r*)<sup>&minus;2</sup> |

## Prerequisites

* **Python 3.8+:** `mpmath`, `numpy`, `matplotlib`
* **C++ Compiler:** `g++` or `clang++` with C++17 support
* **Eigen3:** Header-only C++ linear algebra library (`libeigen3-dev` on Ubuntu/Debian, `eigen` via Homebrew on macOS, or MSYS2/vcpkg on Windows)

## Workflow & Execution Instructions

Move into the desired model directory (`cd spl`, `cd plummer`, or `cd hernquist`) and execute the two stages sequentially.

### Stage 1: Initial Conditions (`initialization.py`)

Run the Python script to solve the spherical Jeans equation at high precision (`mpmath`) and discretize the halo into 180 Lagrangian shells:

```bash
python initialization.py
```

**Generated Outputs:**
Creates `./initial/<tag>/` (by default `./initial/20261007-180layers/`) containing:
* `Basic-<tag>.txt`: Summary of physical and numerical parameters at *t* = 0.
* `RList-<tag>.txt`: Radial shell boundaries (*r*).
* `RhoList-<tag>.txt`: Initial dark matter density profile (*ρ*<sub>&chi;</sub>).
* `MList-<tag>.txt`: Initial enclosed dark matter mass (*M*<sub>&chi;</sub>).
* `uList-<tag>.txt`: Initial specific internal energy (*u*<sub>&chi;</sub> = 1.5 *v*<sub>1D</sub><sup>2</sup>).
* `LList-<tag>.txt`: Initial conductive luminosity profile (*L*<sub>&chi;</sub>).
* `profile.pdf`: Log-log diagnostic plot of the initial radial profiles (*M*<sub>&chi;</sub>, *ρ*<sub>&chi;</sub>, *ρ*<sub>b</sub>, *v*<sub>&chi;</sub>, *L*<sub>&chi;</sub>, and Knudsen number Kn<sub>&chi;</sub>).

### Stage 2: Gravothermal Evolution (`evolution.cpp`)

Ensure the output directory `results/` exists, then compile and run the C++ solver to integrate heat conduction and hydrostatic relaxation over time:

**Linux / macOS:**
```bash
mkdir -p results
g++ -O3 -std=c++17 -I/usr/include/eigen3 evolution.cpp -o evolution
./evolution
```
*(On macOS with Homebrew, replace `/usr/include/eigen3` with `/opt/homebrew/include/eigen3` or `/usr/local/include/eigen3`.)*

**Windows (PowerShell / MSYS2 MinGW64):**
```powershell
New-Item -ItemType Directory -Force -Path results
g++ -O3 -std=c++17 -IC:/msys64/mingw64/include/eigen3 evolution.cpp -o evolution.exe
.\evolution.exe
```

**Generated Outputs:**
Writes `./results/result_<tag>.txt` (by default `./results/result_20261007-180layers.txt`), containing:
* A header block recording all simulation parameters and initial boundary values.
* Time-series snapshots (saved every 100 steps, increasing to every step once *ρ*<sub>c</sub> &ge; 10<sup>6</sup>), each reporting `totalTime` and `step` followed by five rows of radial arrays: `RList`, `RhoList`, `MList`, `uList`, and `LList`.
