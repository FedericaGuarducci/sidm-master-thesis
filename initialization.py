import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


# ==============================================================================
# 1. Configurazione
# ==============================================================================
# Imposta precisione ad alta accuratezza per mpmath
mp.mp.dps = 25

# ----------------------
# USER CONFIGURATION
# ----------------------
base_path = r"C:\Users\feddy\Desktop\FedeFISICA\Gaggero\Programmi"

# Parametri del modello SIDM e barionici
a = mp.mpf('2.257')
b = mp.mpf('1.385')
c = mp.mpf('0.753')

my_mass_norm = mp.mpf('0.1')
my_scale_norm = mp.mpf('0.1')  # Mantenuto per compatibilità di firme
my_sigma = mp.mpf('0.5')

# Griglia lagrangiana 1D
r_min = mp.mpf('0.005')
r_max = mp.mpf('500.0')
layer = 400
extra_layer = 10

my_tag = "20261007-400layers"

# ==============================================================================
# 2. Definizione delle funzioni massa e densità
# ==============================================================================
# Dark Matter (NFW)
def density_dm(r):
    """Dark matter density function, assuming a NFW profile"""
    return 1 / (r * (1 + r)**2)

def mass_dm(r):
    """Dark matter mass function, assuming a NFW profile"""
    return -r / (1 + r) + mp.log(1 + r)

# SPL barioni
def density_baryon(r, mass_norm):
    return mp.mpf('0.07') * (r)**(-mp.mpf('2.4'))

def mass_baryon(r, mass_norm):
    return mass_norm * (r)**(mp.mpf('0.6'))

# Totali (DM + Barioni SPL)
def density_total(r, mass_norm):
    return density_dm(r) + density_baryon(r, mass_norm)

def mass_total(r, mass_norm):
    return mass_dm(r) + mass_baryon(r, mass_norm)

# ==============================================================================
# 3. Equazione di Jeans e Luminosità per SPL
# ==============================================================================
def vd_dm(r_target, mass_norm, ars=None):
    """Calcola v_d(r) integrando numericamente l'equazione di Jeans"""
    r_target = mp.mpf(r_target)
    
    def integrand(r):
        r = mp.mpf(r)
        return density_dm(r) * mass_total(r, mass_norm) / (r**2)
    
    # Integrazione fino a r_max per stabilità e velocità
    integral = mp.quad(integrand, [r_target, r_max])
    rho_target = density_dm(r_target)
    
    v_sq = integral / rho_target
    return mp.sqrt(v_sq)

def big_dev(r, mass_norm, ars=None):
    """
    Derivata d(vd^2)/dr esatta dall'equazione di Jeans per SPL:
    d(vd^2)/dr = -G*M_tot(r)/r^2 - vd^2(r) * (1/rho_dm * d(rho_dm)/dr)
    """
    r_val = mp.mpf(r)
    vd_val = vd_dm(r_val, mass_norm)
    
    # Termine gravitazionale -G * M_tot(r) / r^2
    grav_term = -mass_total(r_val, mass_norm) / (r_val**2)
    
    # Termine d(log rho_dm)/dr per il profilo NFW: -(1 + 3r) / (r*(1+r))
    dlog_rho = -(1 + 3 * r_val) / (r_val * (1 + r_val))
    
    return grav_term - (vd_val**2) * dlog_rho

def luminosity_dm(r, a, b, c, sigma, mass_norm, ars=None):
    """Dark matter luminosity function"""
    r_val = mp.mpf(r)
    
    density = density_dm(r_val)
    vd = vd_dm(r_val, mass_norm)
    bd = big_dev(r_val, mass_norm)
    
    factor = -(3/2) * r_val**2
    numerator = a * b * c * sigma * density * vd**3
    denominator = a * c * sigma**2 * density * vd**2 + b
    
    return factor * (numerator / denominator) * bd

# ==============================================================================
# 4. Creazione della griglia e calcolo vettori
# ==============================================================================
def log_space(start, stop, num):
    start_log = mp.log10(start)
    stop_log = mp.log10(stop)
    step = (stop_log - start_log) / (num - 1)
    return [mp.power(10, start_log + i * step) for i in range(num)]

r_list1 = log_space(r_min, r_max, layer + extra_layer)
r_list2 = [r_list1[0] / 2] + [(r_list1[i-1] + r_list1[i]) / 2 for i in range(1, len(r_list1))]

def calculate_lists():
    m_list = [mass_dm(r) for r in r_list1]
    rho_list = [density_dm(r) for r in r_list2]
    vd_list = [vd_dm(r, my_mass_norm) for r in r_list2]
    u_list = [mp.mpf('1.5') * mp.re(v)**2 for v in vd_list]
    l_list = [luminosity_dm(r, a, b, c, my_sigma, my_mass_norm) for r in r_list1]
    
    # Troncamento ai layer richiesti
    r_list1_trunc = r_list1[:layer]
    r_list2_trunc = r_list2[:layer]
    m_list_trunc = m_list[:layer]
    rho_list_trunc = rho_list[:layer]
    u_list_trunc = u_list[:layer]
    vd_list_trunc = vd_list[:layer]
    l_list_trunc = l_list[:layer]
    
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
# 5. Plotting e Salvataggio Dati
# ==============================================================================
def plot_results(results):
    r1 = np.array([float(mp.re(r)) for r in results['r_list1_trunc']])
    r2 = np.array([float(mp.re(r)) for r in results['r_list2_trunc']])
    m = np.array([float(mp.re(val)) for val in results['m_list_trunc']])
    rho = np.array([float(mp.re(val)) for val in results['rho_list_trunc']])
    rho_b = np.array([float(mp.re(density_baryon(r, my_mass_norm))) for r in r2])
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
    plt.title('SPL Baryon + NFW DM Initial Condition')
    plt.xlabel('r')
    plt.tight_layout()
    
    return plt

def export_data(results, my_tag=None):
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
        "Initial baryon profile = SPL",
        f"rmin = {float(r_min)}",
        f"rmax = {float(r_max)}",
        f"Shell Number = {len(r_list1)}",
        f"Extra shell = {extra_layer}",
        f"baryon_SPL_mass_norm = {float(my_mass_norm)}"
    ]
    
    basic_file = os.path.join(output_dir, f"Basic-{my_tag}.txt")
    with open(basic_file, 'w') as f:
        f.write('\n'.join(basic_info))
    
    r_list_str = [f"{float(mp.re(r)):.10g}" for r in results['r_list1_trunc']]
    m_list_str = [f"{float(mp.re(m)):.10g}" for m in results['m_list_trunc']]
    rho_list_str = [f"{float(mp.re(rho)):.10g}" for rho in results['rho_list_trunc']]
    u_list_str = [f"{float(mp.re(u)):.10g}" for u in results['u_list_trunc']]
    l_list_str = [f"{float(mp.re(l)):.10g}" for l in results['l_list_trunc']]
    
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
# 6. Esecuzione principale
# ==============================================================================
if __name__ == "__main__":
    results = calculate_lists()
    output_dir = export_data(results, my_tag)
    
    plt_fig = plot_results(results)
    pdf_path = os.path.join(output_dir, "profile.pdf")
    plt_fig.savefig(pdf_path)

    print(f"Calcolo ed esportazione completati con successo per SPL in:\n{output_dir}")

