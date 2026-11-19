# Simulation Post-Processing & Diagnostics

Python scripts for analyzing the gravothermal evolution outputs, verifying numerical convergence across grid resolutions, and converting dimensionless profiles into physical astrophysical units.

---

## Requirements

* Python 3.8+
* `numpy`
* `scipy` (for spline interpolation and derivative diagnostics)
* `matplotlib`

---

## Scripts Overview

* **`convergence_layers.py`:** Interpolates density profiles at an exact central density threshold (*ρ*<sub>c</sub> = 10<sup>17</sup>) across runs with different shell counts (*N*<sub>layers</sub>) and computes relative convergence ratios for collapse times and inner core masses (*M*<sub>in</sub> / *M*<sub>0</sub>).
* **`dimensionless_units.py`:** Plots the time evolution of normalized density (*ρ*<sub>&chi;</sub> / *ρ*<sub>s</sub>) versus enclosed mass (*M*<sub>&chi;</sub> / *M*<sub>0</sub>) and identifies the asymptotic core mass at steepest logarithmic slope.
* **`layers_distribution.py`:** Tracks the radial distribution of Lagrangian shells across snapshots and locates the inflection radius (*r*<sub>inflect</sub>) via spline second derivatives.
* **`physical_units.py`:** Converts dimensionless simulation outputs into physical astrophysical units (kpc, M<sub>&odot;</sub>, Myr, *v*/*c*) and computes the radial Knudsen number profile (Kn = &lambda;<sub>mfp</sub> / *H*<sub>Jeans</sub>) using the fiducial scale parameters:
  * **Scale density:** *ρ*<sub>s</sub> = 1.5 &times; 10<sup>8</sup> M<sub>&odot;</sub> / kpc<sup>3</sup>
  * **Scale radius:** *r*<sub>s</sub> = 3.2 kpc
  * **Characteristic mass:** *M*<sub>0</sub> = 4&pi;*ρ*<sub>s</sub>*r*<sub>s</sub><sup>3</sup> &approx; 6.18 &times; 10<sup>10</sup> M<sub>&odot;</sub>
  * **Characteristic velocity:** *v*<sub>s</sub> = &radic;(4&pi;*G*&rho;<sub>s</sub>*r*<sub>s</sub><sup>2</sup>) &approx; 288.2 km/s
  * **Characteristic timescale:** *t*<sub>0</sub> = 108.5 Myr
  * **SIDM cross section per unit mass:** *&sigma;*/*m* = 0.01 cm<sup>2</sup> / g (2.0884 &times; 10<sup>&minus;12</sup> kpc<sup>2</sup> / M<sub>&odot;</sub>)

---

## Usage

Run any diagnostic script from inside `analysis/`. Each script takes an optional path to the simulation output as its first argument; without it, it reads the default location listed below.

| Script | Default input | Optional argument |
| :--- | :--- | :--- |
| `convergence_layers.py` | `../02-nobar-implicit-amr/results/` (all `result_*.txt`) | results directory |
| `layers_distribution.py` | `../02-nobar-implicit-amr/results/result_20261006_180Layers_nobar.txt` | result file |
| `physical_units.py` | `../01-baseline/spl/results/result_20261007-180layers.txt` | result file |
| `dimensionless_units.py` | `../01-baseline/hernquist/results/` (plots the analytical Hernquist baryon model) | results directory |

```bash
cd analysis
python convergence_layers.py                      # uses the default location
python layers_distribution.py path/to/result_<tag>.txt
python physical_units.py path/to/result_<tag>.txt
python dimensionless_units.py path/to/results_dir
```

Figures are written to `analysis/output/` (or next to the script for `convergence_layers.py` if no output directory is given).
