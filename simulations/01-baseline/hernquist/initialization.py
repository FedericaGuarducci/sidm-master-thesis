import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


# ==============================================================================
# 1. Configuration
# ==============================================================================
# Set high numerical precision for mpmath
mp.mp.dps = 25

# ----------------------
# USER CONFIGURATION
# ----------------------
base_path = "."

# SIDM conductivity and baryonic model parameters
# a, b, c are the parameters for the SIDM conductivity terms
a = mp.mpf('2.257')
b = mp.mpf('1.385')
c = mp.mpf('0.753')

# my_mass_norm is the normalized baryon mass, M_b/(4*pi*rho_s*r_s^3)
my_mass_norm = mp.mpf('0.1')
# my_scale_norm is the normalized baryon scale radius, a/r_s
my_scale_norm = mp.mpf('0.1')
# my_sigma is the normalized SIDM cross section (sigma/m)*rho_s*r_s
my_sigma = mp.mpf('0.5')

# 1D Lagrangian grid parameters
r_min = mp.mpf('0.005')  # default 10^-4
r_max = mp.mpf('500.0')  # default 10^2
layer = 180
# extra_layer points are added to ensure a smooth 1D velocity dispersion profile
extra_layer = 10

# Output tag (must match the tag in evolution.cpp)
my_tag = "20261007-180layers"


# ==============================================================================
# 2. Mass and Density Profile Definitions
# ==============================================================================
# Dark Matter (NFW)
def density_dm(r):
    """Dark matter density profile assuming an NFW halo."""
    return 1 / (r * (1 + r)**2)


def mass_dm(r):
    """Enclosed dark matter mass assuming an NFW profile."""
    return -r / (1 + r) + mp.log(1 + r)


# Baryons (Hernquist profile)
def density_baryon(r, mass_norm, ars):
    """
    Baryon density profile assuming a Hernquist sphere.
    mass_norm: normalized baryon mass, M_b/(4*pi*rho_s*r_s^3)
    ars: normalized scale radius, a/r_s
    """
    return 2 * mass_norm * ars * (r * (r + ars)**3)**(-1)


def mass_baryon(r, mass_norm, ars):
    """Enclosed baryon mass assuming a Hernquist profile."""
    return mass_norm * (1 + ars / r)**(-2)


# Total profiles (DM + Hernquist Baryons)
def density_total(r, mass_norm, ars):
    """Total density function (DM + baryons)."""
    return density_dm(r) + density_baryon(r, mass_norm, ars)


def mass_total(r, mass_norm, ars):
    """Total enclosed mass function (DM + baryons)."""
    return mass_dm(r) + mass_baryon(r, mass_norm, ars)


# ==============================================================================
# 3. Jeans Equation and Luminosity for the Hernquist Model
# ==============================================================================
def vd_dm(r_target, mass_norm, ars):
    """Compute dark matter 1D velocity dispersion by numerically integrating the Jeans equation."""
    r_target = mp.mpf(r_target)

    # Jeans integrand: rho_dm(r) * G * M_tot(r) / r^2
    def integrand(r):
        r = mp.mpf(r)
        return density_dm(r) * mass_total(r, mass_norm, ars) / (r**2)

    # Integrate from target radius to infinity
    integral = mp.quad(integrand, [r_target, mp.inf])
    rho_target = density_dm(r_target)

    # v_d^2(r) = (1 / rho_dm(r)) * Integral
    v_sq = integral / rho_target
    return mp.sqrt(v_sq)


def big_dev(r, mass_norm, ars):
    """Exact radial derivative d(vd_dm^2)/dr from the Jeans equation."""
    r_val = mp.mpf(r)
    vd_val = vd_dm(r_val, mass_norm, ars)

    # Gravitational term: -G * M_tot(r) / r^2
    grav_term = -mass_total(r_val, mass_norm, ars) / (r_val**2)

    # Logarithmic density slope term: -v_d^2 * d(ln rho_dm)/dr for NFW
    dlog_rho = -(1 + 3 * r_val) / (r_val * (1 + r_val))

    return grav_term - (vd_val**2) * dlog_rho


def luminosity_dm(r, a, b, c, sigma, mass_norm, ars):
    """Dark matter conductive luminosity function."""
    r_val = mp.mpf(r)

    density = density_dm(r_val)
    vd = vd_dm(r_val, mass_norm, ars)
    bd = big_dev(r_val, mass_norm, ars)

    factor = -(mp.mpf('3') / 2) * r_val**2
    numerator = a * b * c * sigma * density * vd**3
    denominator = a * c * sigma**2 * density * vd**2 + b

    return factor * (numerator / denominator) * bd


# ==============================================================================
# 4. Grid Generation and Vector Evaluation
# ==============================================================================
def log_space(start, stop, num):
    """Create logarithmically spaced radius points."""
    start_log = mp.log10(start)
    stop_log = mp.log10(stop)
    step = (stop_log - start_log) / (num - 1)
    return [mp.power(10, start_log + i * step) for i in range(num)]


# r_list1: shell boundaries; r_list2: shell midpoints
r_list1 = log_space(r_min, r_max, layer + extra_layer)
r_list2 = [r_list1[0] / 2] + [(r_list1[i - 1] + r_list1[i]) / 2 for i in range(1, len(r_list1))]


def calculate_lists():
    """Evaluate radial profiles and truncate to the active number of layers."""
    m_list = [mass_dm(r) for r in r_list1]
    rho_list = [density_dm(r) for r in r_list2]
    vd_list = [vd_dm(r, my_mass_norm, my_scale_norm) for r in r_list2]
    u_list = [mp.mpf('1.5') * mp.re(v)**2 for v in vd_list]
    l_list = [luminosity_dm(r, a, b, c, my_sigma, my_mass_norm, my_scale_norm) for r in r_list1]

    # Truncate extra layers
    r_list1_trunc = r_list1[:layer]
    r_list2_trunc = r_list2[:layer]
    m_list_trunc = m_list[:layer]
    rho_list_trunc = rho_list[:layer]
    u_list_trunc = u_list[:layer]
    vd_list_trunc = vd_list[:layer]
    l_list_trunc = l_list[:layer]

    # Calculate Knudsen number
    kn_list_trunc = [
        (1 / (my_sigma * rho_list[i])) / mp.sqrt((2 * u_list[i]) / (3 * rho_list[i]))
        for i in range(layer)
    ]

    return {
        'r_list1_trunc': r_list1_trunc,
        'r_list2_trunc': r_list2_trunc,
        'm_list_trunc': m_list_trunc,
        'rho_list_trunc': rho_list_trunc,
        'u_list_trunc': u_list_trunc,
        'vd_list_trunc': vd_list_trunc,
        'l_list_trunc': l_list_trunc,
        'kn_list_trunc': kn_list_trunc
    }


# ==============================================================================
# 5. Plotting and Data Export
# ==============================================================================
def plot_results(results):
    """Plot initial radial profiles."""
    r1 = np.array([float(mp.re(r)) for r in results['r_list1_trunc']])
    r2 = np.array([float(mp.re(r)) for r in results['r_list2_trunc']])
    m = np.array([float(mp.re(val)) for val in results['m_list_trunc']])
    rho = np.array([float(mp.re(val)) for val in results['rho_list_trunc']])
    rho_b = np.array([float(mp.re(density_baryon(r, my_mass_norm, my_scale_norm))) for r in r2])
    vd = np.array([float(mp.re(val)) for val in results['vd_list_trunc']])
    lum = np.array([float(mp.re(val)) for val in results['l_list_trunc']])
    kn = np.array([float(mp.re(val)) for val in results['kn_list_trunc']])

    plt.figure(figsize=(10, 10))

    plt.loglog(r1, m, label=r'$M_{\chi}$')
    plt.loglog(r2, rho, label=r'$\rho_{\chi}$')
    plt.loglog(r2, rho_b, label=r'$\rho_{b}$')
    plt.loglog(r2, vd, label=r'$\nu_{\chi}$')
    plt.loglog(r1, lum, label=r'$L_{\chi}$')
    plt.loglog(r1, -lum, label=r'$-L_{\chi}$')
    plt.loglog(r2, kn, label=r'$Kn_{\chi}$')

    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.title('Hernquist Baryon + NFW DM Initial Condition')
    plt.xlabel('r')
    plt.tight_layout()

    return plt


def export_data(results, my_tag=None):
    """Export initial profiles to text files for evolution.cpp."""
    if my_tag is None:
        my_tag = datetime.now().strftime("%Y%m%d") + "A"

    initial_dir = os.path.join(base_path, "initial")
    os.makedirs(initial_dir, exist_ok=True)

    output_dir = os.path.join(initial_dir, my_tag)
    os.makedirs(output_dir, exist_ok=True)

    basic_info = [
        f"name = {my_tag}",
        "t = 0",
        f"a = {a}",
        f"b = {b}",
        f"c = {c}",
        f"sigma = {my_sigma}",
        "Initial dark matter profile = NFW",
        "Initial baryon profile = Hernquist",
        f"rmin = {float(r_min)}",
        f"rmax = {float(r_max)}",
        f"Shell Number = {len(r_list1)}",
        f"Extra shell = {extra_layer}",
        f"baryon_Hernquist_mass_norm = {float(my_mass_norm)}",
        f"baryon_Hernquist_ars = {float(my_scale_norm)}"
    ]

    basic_file = os.path.join(output_dir, f"Basic-{my_tag}.txt")
    with open(basic_file, 'w') as f:
        f.write('\n'.join(basic_info))

    # Format values with 10 significant digits
    r_list_str = [f"{float(mp.re(r)):.10g}" for r in results['r_list1_trunc']] + ['']
    m_list_str = [f"{float(mp.re(m)):.10g}" for m in results['m_list_trunc']] + ['']
    rho_list_str = [f"{float(mp.re(rho)):.10g}" for rho in results['rho_list_trunc']] + ['']
    u_list_str = [f"{float(mp.re(u)):.10g}" for u in results['u_list_trunc']] + ['']
    l_list_str = [f"{float(mp.re(l)):.10g}" for l in results['l_list_trunc']] + ['']

    with open(os.path.join(output_dir, f"RList-{my_tag}.txt"), 'w') as f:
        f.write('\n'.join(r_list_str))

    with open(os.path.join(output_dir, f"MList-{my_tag}.txt"), 'w') as f:
        f.write('\n'.join(m_list_str))

    with open(os.path.join(output_dir, f"RhoList-{my_tag}.txt"), 'w') as f:
        f.write('\n'.join(rho_list_str))

    with open(os.path.join(output_dir, f"uList-{my_tag}.txt"), 'w') as f:
        f.write('\n'.join(u_list_str))

    with open(os.path.join(output_dir, f"LList-{my_tag}.txt"), 'w') as f:
        f.write('\n'.join(l_list_str))

    return output_dir


# ==============================================================================
# 6. Main Execution
# ==============================================================================
if __name__ == "__main__":
    results = calculate_lists()
    output_dir = export_data(results, my_tag)

    plt_fig = plot_results(results)
    pdf_path = os.path.join(output_dir, "profile.pdf")
    plt_fig.savefig(pdf_path)

    print(f"Calculation and export completed successfully for the Hernquist profile in:\n{output_dir}")
