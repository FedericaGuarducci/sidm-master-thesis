import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


# ==============================================================================
# 1. Configuration
# ==============================================================================

# Set precision 
mp.mp.dps = 25

# ----------------------
# USER CONFIGURATION
# ----------------------
# Absolute path where output directories should be created
# Change this to your desired output location
base_path = r"."

# Model parameters
# a,b,c are the parameters for the SIDM conductivity terms
a = mp.mpf('2.257')
b = mp.mpf('1.385')
c = mp.mpf('0.753')
# my_sigma is the normalized SIDM cross section (sigma/m)*rho_s*r_s
my_sigma = mp.mpf('0.002')

# 1D Lagrangian zone parameters
r_min = mp.mpf('0.0000000005')  # default 10^-4
r_max = mp.mpf('500.0')  # default 10^2
layer = 180
# extra_layers are added to the end of the list to ensure a smooth 1D velocity dispersion profile
extra_layer = 10

# Output name

my_tag = "20261006_180Layers_nobar"

# ==============================================================================
# 2. Definition of mass and density functions
# ==============================================================================
#%%Dark matter
def density_dm(r):
    """Dark matter density function, assuming a NFW profile"""
    return 1/(r * (1+r)**2)

def mass_dm(r):
    """Dark matter mass function, assuming a NFW profile"""
    return -r/(1+r) + mp.log(1+r)


#%%Velocity calculated with SPL
def vd_dm(r_target):
    """Computes v_d(r) by numerically integrating the Jeans equation"""
    r_target = mp.mpf(r_target)
    
    # Jeans integrand: rho_dm(r) * G * M_b(r) / r^2
    def integrand(r):
        r = mp.mpf(r)
        return density_dm(r) * mass_dm(r) / (r**2)
    
    # Integrate from the target radius to the maximum box radius (r_max = 500)
    integral = mp.quad(integrand, [r_target, mp.inf])
    rho_target = density_dm(r_target)
    
    # v_d^2(r) = (1 / rho(r)) * Integral
    v_sq = integral / rho_target
    return mp.sqrt(v_sq)

def big_dev(r):
    """deviation function of partial vd_dm^2/partial r"""
    r = mp.mpf(r)
    
    result = 1/2 * (
        -4 + mp.pi**2 - 4*r + 4*mp.pi**2*r + 5*r**2 + 3*mp.pi**2*r**2 - 
        8/(1+r) - (26*r)/(1+r) - (22*r**2)/(1+r) - (5*r**3)/(1+r) + 
        1/(r+r**2) + 
        6*mp.log(1+1/r) + 24*r*mp.log(1+1/r) + 18*r**2*mp.log(1+1/r) + 
        5*mp.log(r) + 20*r*mp.log(r) + 15*r**2*mp.log(r) - 
        20*mp.log(1+r) - mp.log(1+r)/r**2 - 38*r*mp.log(1+r) - 15*r**2*mp.log(1+r) + 
        3*mp.log(1+r)**2 + 12*r*mp.log(1+r)**2 + 9*r**2*mp.log(1+r)**2 + 
        6 * (1+4*r+3*r**2) * mp.polylog(2, -r)
    )
    
    return result

def luminosity_dm(r, a, b, c, sigma):
    """Dark matter luminosity function"""
    r_val = mp.mpf(r)
    
    density = density_dm(r_val)
    vd = vd_dm(r_val)
    bd = big_dev(r_val)
    
    factor = -(3/2) * r_val**2
    numerator = a * b * c * sigma * density * vd**3
    denominator = a * c * sigma**2 * density * vd**2 + b
    
    return factor * (numerator / denominator) * bd

# ==============================================================================
# 3. Grid creation
# ==============================================================================
# Create logarithmically spaced radius points
def log_space(start, stop, num):
    """Create logarithmically spaced points similar to Mathematica's Subdivide"""
    start_log = mp.log10(start)
    stop_log = mp.log10(stop)
    step = (stop_log - start_log) / (num - 1)
    return [mp.power(10, start_log + i * step) for i in range(num)]

# Create radius lists
# r_list1 is based on the radius range and number of layers
r_list1 = log_space(r_min, r_max, layer + extra_layer)
# r_list2 is based on the average of adjacent points in r_list1
r_list2 = [r_list1[0] / 2] + [(r_list1[i-1] + r_list1[i]) / 2 for i in range(1, len(r_list1))]

# ==============================================================================
# 4. List calculation
# ==============================================================================
# Calculate lists
def calculate_lists():
    # Calculate the dark matter enclosed mass
    m_list = [mass_dm(r) for r in r_list1]
    # Calculate the dark matter density
    rho_list = [density_dm(r) for r in r_list2]
    # Calculate the dark matter 1D velocity dispersion
    vd_list = [vd_dm(r) for r in r_list2]
    # Calculate specific kinetic energy 
    u_list = [mp.mpf('1.5') * mp.re(v)**2 for v in vd_list]
    # Calculate the dark matter luminosity
    l_list = [luminosity_dm(r, a, b, c, my_sigma) for r in r_list1]
    
    # Truncate to extra layers
    r_list1_trunc = r_list1[:layer]
    r_list2_trunc = r_list2[:layer]
    m_list_trunc = m_list[:layer]
    rho_list_trunc = rho_list[:layer]
    u_list_trunc = u_list[:layer]
    vd_list_trunc = vd_list[:layer]
    l_list_trunc = l_list[:layer]
    
    # Calculate Knudsen number
    kn_list_trunc = [(1/(my_sigma * rho_list[i])) / 
                     mp.sqrt((2 * u_list[i]) / (3 * rho_list[i])) 
                     for i in range(layer)]
    
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
# 5. Definition of plotting and data saving functions
# ==============================================================================
# Plotting of results
def plot_results(results):
    """Plot the results similar to the Mathematica plot"""
    # Convert mpmath objects to numpy floats for plotting (handling complex numbers)
    r1 = np.array([float(mp.re(r)) for r in results['r_list1_trunc']])
    r2 = np.array([float(mp.re(r)) for r in results['r_list2_trunc']])
    m = np.array([float(mp.re(val)) for val in results['m_list_trunc']])
    rho = np.array([float(mp.re(val)) for val in results['rho_list_trunc']])
    vd = np.array([float(mp.re(val)) for val in results['vd_list_trunc']])
    lum = np.array([float(mp.re(val)) for val in results['l_list_trunc']])
    kn = np.array([float(mp.re(val)) for val in results['kn_list_trunc']])
    
    plt.figure(figsize=(10, 10))
    
    plt.loglog(r1, m, label=r'$M_{\chi}$')
    plt.loglog(r2, rho, label=r'$\rho_{\chi}$')
    plt.loglog(r2, vd, label=r'$\nu_{\chi}$')
    plt.loglog(r1, lum, label=r'$L_{\chi}$')
    plt.loglog(r1, -lum, label=r'$-L_{\chi}$')
    plt.loglog(r2, kn, label=r'$Kn_{\chi}$')
    
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.title('Red Dot Halo Initial Condition')
    plt.xlabel('r')
    plt.tight_layout()
    
    return plt

# Data export
def export_data(results, my_tag=None):
    """Export data to files similar to the Mathematica export"""
    if my_tag is None:
        my_tag = datetime.now().strftime("%Y%m%d") + "A"
    
    # Use the base_path defined at the top of the script
    # First check if "initial" folder exists, create it if not
    initial_dir = os.path.join(base_path, "initial")
    os.makedirs(initial_dir, exist_ok=True)
    
    # Create the date directory inside the initial folder
    output_dir = os.path.join(initial_dir, my_tag)
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic parameters with default formatting
    basic_info = [
        f"name = {my_tag}",
        "t = 0",
        f"a = {a}",
        f"b = {b}",
        f"c = {c}",
        f"sigma = {my_sigma}",
        "Initial dark matter profile = NFW",
        "Initial baryon profile = none",
        f"rmin = {float(r_min)}",
        f"rmax = {float(r_max)}",
        f"Shell Number = {len(r_list1)}",
        f"Extra shell = {extra_layer}",
    ]
    
    # Write basic info
    basic_file = os.path.join(output_dir, f"Basic-{my_tag}.txt")
    with open(basic_file, 'w') as f:
        f.write('\n'.join(basic_info))
    
    # Convert mpmath values to strings for export with 10 effective digits
    # Use mp.re to extract real part if values are complex
    r_list_str = [f"{float(mp.re(r)):.10g}" for r in results['r_list1_trunc']] + ['']
    m_list_str = [f"{float(mp.re(m)):.10g}" for m in results['m_list_trunc']] + ['']
    rho_list_str = [f"{float(mp.re(rho)):.10g}" for rho in results['rho_list_trunc']] + ['']
    u_list_str = [f"{float(mp.re(u)):.10g}" for u in results['u_list_trunc']] + ['']
    l_list_str = [f"{float(mp.re(l)):.10g}" for l in results['l_list_trunc']] + ['']
    
    # Write data files with full paths
    r_file = os.path.join(output_dir, f"RList-{my_tag}.txt")
    with open(r_file, 'w') as f:
        f.write('\n'.join(r_list_str))
    
    m_file = os.path.join(output_dir, f"MList-{my_tag}.txt")
    with open(m_file, 'w') as f:
        f.write('\n'.join(m_list_str))
    
    rho_file = os.path.join(output_dir, f"RhoList-{my_tag}.txt")
    with open(rho_file, 'w') as f:
        f.write('\n'.join(rho_list_str))
    
    u_file = os.path.join(output_dir, f"uList-{my_tag}.txt")
    with open(u_file, 'w') as f:
        f.write('\n'.join(u_list_str))
    
    l_file = os.path.join(output_dir, f"LList-{my_tag}.txt")
    with open(l_file, 'w') as f:
        f.write('\n'.join(l_list_str))
    
    return output_dir

# ==============================================================================
# 6. Main execution
# ==============================================================================
# Main execution
if __name__ == "__main__":
    # Calculate all lists
    results = calculate_lists()
    
    # Export data to files and get the output directory path
    output_dir = export_data(results, my_tag)
    
    # Plot and save figure as PDF
    plt_fig = plot_results(results)
    # Save PDF to the same directory as other files
    pdf_path = os.path.join(output_dir, "profile.pdf")
    plt_fig.savefig(pdf_path)

    
    print(f"Calculation and export completed successfully.")
