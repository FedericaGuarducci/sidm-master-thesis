import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


# ==============================================================================
# 1. Configurazione
# ==============================================================================
#482 R(0) is negative
# Set precision 
mp.mp.dps = 25

# ----------------------
# USER CONFIGURATION
# ----------------------
# Absolute path where output directories should be created
# Change this to your desired output location
base_path = r"C:\Users\feddy\Desktop\FedeFISICA\Gaggero\Programmi\8-ProfiliBarioniciDiversi\Hernquist"

# Model parameters
# a,b,c are the parameters for the SIDM conductivity terms
a = mp.mpf('2.257')
b = mp.mpf('1.385')
c = mp.mpf('0.753')
# my_mass_norm is the normalized baryon mass, M_b/(4*pi*rho_s*r_s^3)
my_mass_norm = mp.mpf('0.1')
# my_scale_norm is the normalized baryon scale radius, a/r_s
my_scale_norm = mp.mpf('0.1')
# my_sigma is the normalized SIDM cross section (sigma/m)*rho_s*r_s
my_sigma = mp.mpf('0.5')

# 1D Lagragian zone parameters
r_min = mp.mpf('0.005')  # default 10^-4
r_max = mp.mpf('500.0')  # default 10^2
layer = 180
# extra_layers are added to the end of the list to ensure a smooth 1D velocity dispersion profile
extra_layer = 10

# Output name

my_tag = "20261007-180layers"

# ==============================================================================
# 2. Definizione delle funzioni massa e densità
# ==============================================================================
#%%Dark matter
def density_dm(r):
    """Dark matter density function, assuming a NFW profile"""
    return 1/(r * (1+r)**2)

def mass_dm(r):
    """Dark matter mass function, assuming a NFW profile"""
    return -r/(1+r) + mp.log(1+r)

#%%Plummer barioni(this is not the profile used in the article)
#def density_baryon(r, mass_norm, ars):
    #"""Baryon density function, assuming a Plummer profile"""
    #"""mass_norm is the normalized baryon mass, M_b/(4*pi*rho_s*r_s^3)"""
    #"""ars is the normalized scale radius, a/r_s"""
    #return (3*mass_norm)/(ars**3) * (1+r**2/ars**2)**(-5/2)

#def mass_baryon(r, mass_norm, ars):
   # """Baryon mass function, assuming a Plummer profile"""
   # return mass_norm * (1 + ars**2 * r**(-2))**(-1.5)
   
#%%SPL barioni
#def density_baryon(r, mass_norm):
#    return 0.07*(r)**(-2.4)

#def mass_baryon(r, mass_norm):
#    return mass_norm*(r)**(0.6)
    
#%% Hernquist barioni
def density_baryon(r, mass_norm, ars):
    return 2*mass_norm*ars*(r*(r+ars)**3)**(-1)

def mass_baryon(r, mass_norm, ars):
    return mass_norm*(1+ ars/r)**(-2)
    
    
#%%Totali
def density_total(r, mass_norm, ars):
    """Total density function"""
    return density_dm(r) + density_baryon(r, mass_norm, ars)

def mass_total(r, mass_norm, ars):
    """Total enclosed mass function"""
    return mass_dm(r) + mass_baryon(r, mass_norm, ars)

#%%Velocità calcolata con Plummer
#def vd_dm(r, mass_norm, ars):
    #"""Dark matter 1D velocity dispersion"""
    #r, mass_norm, ars = mp.mpf(r), mp.mpf(mass_norm), mp.mpf(ars)
    
    #term = -(1/2) * r * (
    #    (2 * mass_norm * (1+r) * (
    #        mp.sqrt(1+ars**2) * (
    #            -ars**4 - (1+r) * (-r + mp.sqrt(ars**2 + r**2)) + 
    #            ars**2 * (2+r-2*r**2 + 2*mp.sqrt(ars**2 + r**2) + 2*r*mp.sqrt(ars**2 + r**2))
    #        ) - 
    #        6 * ars**2 * (1+r) * mp.sqrt(ars**2 + r**2) * (
    #            mp.acoth(mp.sqrt(1+ars**2)) + 
    #            mp.atanh((-1-r+mp.sqrt(ars**2 + r**2))/mp.sqrt(1+ars**2))
    #        )
    #    )) / (ars**2 * (1+ars**2)**(5/2) * mp.sqrt(ars**2 + r**2))
    #    +
    #    (2 + 9*r + 6*r**2 - 6*r * (1+r)**2 * mp.log(1+1/r)) / r
    #+
    #(1/r**2) * (1+r) * (
    #       -r * (1 + r*(-1 + mp.pi**2 * (1+r)) + 5*r*(1+r)*mp.log(r)) + 
    #        (-1 + r*(3 + r*(11 + 5*r))) * mp.log(1+r) - 
    #        3 * r**2 * (1+r) * mp.log(1+r)**2 - 
    #        6 * r**2 * (1+r) * mp.polylog(2, -r)
    #    )
    #)
    
    # Ensure we get a real result (in case of small numerical errors)
    #if mp.im(term) != 0 and abs(mp.im(term)) < 1e-10:
    #    term = mp.re(term)
    
    #result = mp.sqrt(term)
    #return result

#%%Velocità calcolata con SPL
#def vd_dm(r_target, mass_norm, ars):
#    """Calcola v_d(r) integrando numericamente l'equazione di Jeans"""
#    r_target = mp.mpf(r_target)
    
    # Integrando di Jeans: rho_dm(r) * G * M_tot(r) / r^2
#    def integrand(r):
#        r = mp.mpf(r)
#        return density_dm(r) * mass_total(r, mass_norm, ars) / (r**2)
    
    # Integriamo dal raggio target al raggio massimo del box (r_max = 500)
#    integral = mp.quad(integrand, [r_target, mp.inf])
#    rho_target = density_dm(r_target)
    
    # v_d^2(r) = (1 / rho(r)) * Integrale
#    v_sq = integral / rho_target
#    return mp.sqrt(v_sq)
    
#%%Velocità calcolata con Hernquist
def vd_dm(r_target, mass_norm, ars):
    """Calcola v_d(r) integrando numericamente l'equazione di Jeans"""
    r_target = mp.mpf(r_target)
    
    # Integrando di Jeans: rho_dm(r) * G * M_tot(r) / r^2
    def integrand(r):
        r = mp.mpf(r)
        return density_dm(r) * mass_total(r, mass_norm, ars) / (r**2)
    
    # Integriamo dal raggio target al raggio massimo del box (r_max = 500)
    integral = mp.quad(integrand, [r_target, mp.inf])
    rho_target = density_dm(r_target)
    
    # v_d^2(r) = (1 / rho(r)) * Integrale
    v_sq = integral / rho_target
    return mp.sqrt(v_sq)

def big_dev(r, mass_norm, ars):
    """Derivata numerica d(vd^2)/dr esatta dall'equazione di Jeans"""
    r_val = mp.mpf(r)
    vd_val = vd_dm(r_val, mass_norm, ars)
    
    # -G * M_tot(r) / r^2
    grav_term = -mass_total(r_val, mass_norm, ars) / (r_val**2)
    
    # - v_d^2 * (1/rho_dm * d(rho_dm)/dr) per NFW
    dlog_rho = -(1 + 3*r_val) / (r_val * (1 + r_val))
    
    return grav_term - (vd_val**2) * dlog_rho

def luminosity_dm(r, a, b, c, sigma, mass_norm, ars):
    """Dark matter luminosity function"""
    r_val = mp.mpf(r)
    
    density = density_dm(r_val)
    vd = vd_dm(r_val, mass_norm, ars)
    bd = big_dev(r_val, mass_norm, ars)
    
    factor = -(3/2) * r_val**2
    numerator = a * b * c * sigma * density * vd**3
    denominator = a * c * sigma**2 * density * vd**2 + b
    
    return factor * (numerator / denominator) * bd

# ==============================================================================
# 3. Creazione della griglia
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
# 4. Calcolo delle liste
# ==============================================================================
# Calculate lists
def calculate_lists():
    # Calculate the dark matter enclosed mass
    m_list = [mass_dm(r) for r in r_list1]
    # Calculate the dark matter density
    rho_list = [density_dm(r) for r in r_list2]
    # Calculate the dark matter 1D velocity dispersion
    vd_list = [vd_dm(r, my_mass_norm, my_scale_norm) for r in r_list2]
    # Calculate specific kinetic energy 
    u_list = [mp.mpf('1.5') * mp.re(v)**2 for v in vd_list]
    # Calculate the dark matter luminosity
    l_list = [luminosity_dm(r, a, b, c, my_sigma, my_mass_norm, my_scale_norm) for r in r_list1]
    
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
# 5. Definizione delle funzioni di plotting e di salvataggio dati
# ==============================================================================
# Plotting of results
def plot_results(results):
    """Plot the results similar to the Mathematica plot"""
    # Convert mpmath objects to numpy floats for plotting (handling complex numbers)
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
    plt.title('Red Dot Halo Initial Condition')
    plt.xlabel('r')
    plt.tight_layout()
    
    return plt

#Data exportation
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
        "Initial baryon profile = Plummer",
        f"rmin = {float(r_min)}",
        f"rmax = {float(r_max)}",
        f"Shell Number = {len(r_list1)}",
        f"Extra shell = {extra_layer}",
        f"baryon_Plummer_mass_norm = {float(my_mass_norm)}",
        f"baryon_Plummer_ars = {float(my_scale_norm)}"
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
# 6. Esecuzione principale
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
