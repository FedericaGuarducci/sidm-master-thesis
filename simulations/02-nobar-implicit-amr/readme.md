# SIDM Gravothermal Evolution (No Baryons, Crank-Nicolson & Adaptive Mesh Refinement)

1D Lagrangian code for simulating the gravothermal evolution of a pure Self-Interacting Dark Matter (SIDM) halo (without baryons). This implementation solves the heat conduction step implicitly via a Crank-Nicolson scheme and enforces hydrostatic equilibrium via an exact tridiagonal Newton-Raphson solver with dynamic layer splitting and coarsening.

---

## Repository Structure

* `initialization.py`  — Script to generate initial NFW profiles (`RList`, `RhoList`, `MList`, `uList`, `LList`, `Basic`) using arbitrary-precision arithmetic (`mpmath`).
* `evolution.cpp` — Main C++17 simulation code.

---

## Compilation & Usage


### Initial Conditions Generator
```bash
python generate_initial.py
```

### C++ Evolution Code
Compile with `g++` (C++17 standard required; **do not** use `-ffast-math` or `-Ofast` so `NaN`/`inf` checks remain valid):

```bash
g++ -O3 -std=c++17 -march=native -fno-math-errno -DNDEBUG evolutionCN-nobar8-2OK.cpp -o evolve
```

Run with default parameters or pass custom arguments via CLI:
```bash
./evolve [tag] [inputDir] [outputFile] [totalStep] [adaptGap] [epsFine] [saveDlog10]
```


---

## Summary of Modifications from the Original Code

### 1. Physics Modifications
| Feature | Original Code | New Code | Reason |
| :--- | :--- | :--- | :--- |
| **Baryonic profile** | SPL baryonic profile ρ<sub>b</sub>(r) = ρ<sub>b,s</sub> (r / r<sub>s</sub>)<sup>-2.4</sup> | No baryons (dark matter only) | Code simplification |
| **Normalized cross section** | σ = 0.5 | σ = 0.001 *(default in C++)* | Different assumptions across reference papers |

---

### 2. Numerical & Algorithmic Efficiency
| Feature | Original Code | New Code | Reason |
| :--- | :--- | :--- | :--- |
| **Data structures & solvers** | `Eigen` dense/sparse vectors and matrices (𝒪(N³)) | Standard `std::vector<double>` exploiting tridiagonal matrix structure (𝒪(N)) | Significant speedup; removes external `Eigen` dependency |
| **Memory allocation** | Dynamic allocations at every step | Pre-allocated work buffers reused across steps | Faster execution |
| **Math & I/O optimizations** | `std::pow` and `iostream` formatting | Fractional powers via `std::cbrt`, cached entropy `s.a`, and `std::to_chars` output (10-digit precision instead of 17) | Faster runtime and smaller output files |
| **Save frequency** | Every 100 steps initially; every 1 step once ρ₀ / ρ<sub>s</sub> > 10⁶ | Every 500 steps initially; once ρ₀ / ρ<sub>s</sub> > 10⁶, saved every Δlog₁₀(ρ₀) = 0.05 dex (bounded between 100 and 20,000 steps) + guaranteed final snapshot | Faster I/O and drastically smaller output files |

---

### 3. Dynamic Layering (Adaptive Mesh Refinement)
| Feature | Original Code | New Code | Reason |
| :--- | :--- | :--- | :--- |
| **Grid type** | Fixed Lagrangian grid | Dynamic Lagrangian grid (checked every 50 steps) | Better resolution of the inner asymptotic region during core collapse |
| **Splitting criterion** | None | Repeated passes splitting any shell with ln(Rᵢ / Rᵢ₋₁) > 0.15 | Resolves regions where radial gradients become steep |
| **New layer placement** | — | Splits the shell into two equal-mass halves (mᵢ / 2) at R<sub>new</sub> = ((Rᵢ³ + Rᵢ₋₁³) / 2)¹ᐟ³, conserving mass, internal energy, and entropy | Exact conservation laws without introducing large mass jumps |
| **Coarsening (Merging)** | Absent | Merges adjacent shell pairs when ln(R<sub>i+1</sub> / R<sub>i−1</sub>) ≤ 0.65 × 0.15 = 0.0975 and ln(M<sub>i+1</sub> / M<sub>i−1</sub>) ≤ max<sub>k</sub> ln(M<sub>k</sub> / M<sub>k−1</sub>) (from the initial grid), provided entropy monotonicity is not worsened | Prevents over-resolving re-compressed regions, protects the core, and keeps output sizes compact |

---

### 4. Conduction Step Modifications
| Feature | Original Code | New Code | Reason |
| :--- | :--- | :--- | :--- |
| **Time discretization** | Explicit | Implicit Crank-Nicolson solved via the Thomas algorithm | Unconditional numerical stability and larger allowable timesteps |
| **Timestep formula & rollback** | Δt = ε / maxᵢ │(1 / uᵢ) · (Lᵢ − Lᵢ₋₁) / (Mᵢ − Mᵢ₋₁)│ | Δt = s · ε / maxᵢ │(1 / uᵢ) · (Lᵢ − Lᵢ₋₁) / (Mᵢ − Mᵢ₋₁)│ with adaptive scaling s ≤ 1; if a new entropy inversion occurs or relaxation fails to converge, the step is rolled back and s is halved | Prevents unphysical entropy inversions and eases hydrostatic convergence |
| **ε parameter** | Constant ε = 0.001 | ε = 0.003 initially, increased to ε = 0.01 once ρ₀ / ρ<sub>s</sub> ≥ 10⁵ | Speeds up late-stage collapse without shifting the collapse timescale t<sub>s</sub> |

---

### 5. Hydrostatic Relaxation Step Modifications
| Feature | Original Code | New Code | Reason |
| :--- | :--- | :--- | :--- |
| **Linear system solver** | Cholesky factorization (`LDLT`, assuming symmetric positive-definite matrix) | In-place tridiagonal LU factorization with partial pivoting + diagonal equilibration | The exact hydrostatic Jacobian matrix is **not** symmetric |
| **Radii monotonicity check** | Absent | Backtracking line search: halves ΔR until Rᵢ > Rᵢ₋₁ > 0 and R<sub>N-1</sub> < R<sub>N</sub> | Guarantees physical shell ordering and positive volumes |
| **Iteration count** | Fixed 10 Newton iterations | Dynamic stopping once maxᵢ │ΔRᵢ / Rᵢ│ < 10⁻¹⁰ (up to 30 iterations; triggers timestep rollback if unconverged) | Exploits quadratic convergence to avoid redundant iterations |
| **ρ and p updates** | First-order linearization in ΔR | Exact adiabatic volume scaling: ρ<sub>new</sub>= ρ<sub>old</sub> (V<sub>old</sub> / V<sub>new</sub>), p<sub>new</sub> = p<sub>old</sub> (V<sub>old</sub> / V<sub>new</sub>)⁵ᐟ³ | Higher accuracy and exact adiabatic conservation during large steps |

---

## Key Validation Results
* **Layer independence:** Simulations initialized with 180 to 1000 layers converge to the same collapse time (t / t₀ ≈ 2.04 × 10⁵), core collapse mass (M<sub>in</sub> / M₀ ≈ 2.64 × 10⁻⁵), and density profile ρ(M) at ρ<sub>center</sub> = 10¹⁷ ρ<sub>s</sub>.
* **Asymptotic resolution:** Dynamic splitting and coarsening maintain uniform logarithmic coverage across the entire radial range (10⁻¹² ≤ r / r<sub>s</sub> ≤ 10³) throughout core collapse.
