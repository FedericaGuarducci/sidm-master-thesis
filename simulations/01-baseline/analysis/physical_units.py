from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ==============================================================================
# GLOBAL LATEX RENDERING CONFIGURATION
# ==============================================================================
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "font.size": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ==============================================================================
# 1. FILE PATHS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
filepath = (BASE_DIR / ".." / "spl" / "results" / "result_20261007-180layers.txt").resolve()
output_dir = (BASE_DIR / "output").resolve()
output_pdf_path = output_dir / f"{filepath.stem}_physical_plots.pdf"

# ==============================================================================
# 2. CONSTANTS & SCALE PARAMETERS FOR J1205-0000 HALO
# ==============================================================================
c_kms = 299792.458          # Speed of light [km/s]
G_kpc = 4.30091e-6          # Gravitational constant [kpc (km/s)^2 / M_sun]

# Scale parameters
rho_s = 1.5e8               # M_sun / kpc^3
r_s   = 3.2                 # kpc
v_s   = np.sqrt(4.0 * np.pi * G_kpc * rho_s * (r_s**2))  # ~ 288.2 km/s (natural velocity unit)
t_0   = 108.5               # Myr
sigma_m = 0.01              # cm^2 / g

# Cross-section unit conversion: 1 cm^2/g = 2.0884e-10 kpc^2 / M_sun
sigma_m_kpc2_Msun = sigma_m * 2.0884e-10

# Characteristic mass scale M_0 = 4 * pi * rho_s * r_s^3
M_0 = 4.0 * np.pi * rho_s * (r_s**3)


def parse_simulation_output(file_path):
    """
    Reads and parses the output text file line-by-line to prevent MemoryError.
    """
    snapshots = []
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    def read_vec(f):
        return np.array(f.readline().split(), dtype=float)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            line = f.readline()
            if not line:
                break  # End of file

            line_str = line.strip()

            # Skip blank or header lines
            if not line_str or line_str.startswith((
                "Input", "Initial", "Number", "Total",
                "Save", "Abs", "Cross", "Conduction",
                "Baryon", "time,"
            )):
                continue

            parts = line_str.split()
            # Time/step line contains 2 entries
            if len(parts) == 2:
                try:
                    t_code = float(parts[0])
                    step = int(parts[1])

                    # Read the 5 consecutive profile vectors
                    R = read_vec(f)
                    Rho = read_vec(f)
                    M = read_vec(f)
                    u = read_vec(f)
                    L = read_vec(f)

                    snapshots.append({
                        "t_code": t_code,
                        "t_Myr": t_code * t_0,
                        "step": step,
                        "R": R,
                        "Rho": Rho,
                        "M": M,
                        "u": u,
                        "L": L,
                    })
                except (ValueError, IndexError):
                    pass

    return snapshots


def compute_knudsen(rho_phys, v_1d_kms):
    """
    Knudsen number Kn = lambda_mfp / H, with
      lambda_mfp = 1 / (rho * sigma/m)               [kpc]
      H          = sigma_1D / sqrt(4 pi G rho)       [kpc]  (Jeans scale)
    No radial derivatives: no numerical artefacts in flat cores.
    """
    lambda_mfp = 1.0 / (rho_phys * sigma_m_kpc2_Msun)
    H_jeans = v_1d_kms / np.sqrt(4.0 * np.pi * G_kpc * rho_phys)
    return lambda_mfp / H_jeans


def generate_plots(input_file, save_path, num_profiles=7):
    save_path = Path(save_path)
    snapshots = parse_simulation_output(input_file)
    if not snapshots:
        print("No profiles extracted from the results file.")
        return

    num_valid = len(snapshots)

    # Equally spaced indices by snapshot number
    selected_indices = np.linspace(0, num_valid - 1, num_profiles, dtype=int)
    selected_indices = sorted(list(set(selected_indices)))
    selected_snaps = [snapshots[idx] for idx in selected_indices]

    # Initialize 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    ax_rho, ax_v = axes[0, 0], axes[0, 1]
    ax_m, ax_kn = axes[1, 0], axes[1, 1]

    colors = plt.cm.coolwarm(np.linspace(0, 0.85, len(selected_snaps)))

    for snap, color in zip(selected_snaps, colors):
        # Conversions to physical units
        r_kpc = snap["R"] * r_s                       # kpc
        rho_phys = snap["Rho"] * rho_s                # M_sun / kpc^3
        m_phys = snap["M"] * M_0                      # M_sun

        # 1D velocity dispersion [km/s] and v/c
        v_1d_kms = np.sqrt(2.0 / 3.0 * snap["u"]) * v_s
        v_over_c = v_1d_kms / c_kms

        # Knudsen number with Jeans scale height
        Knudsen = compute_knudsen(rho_phys, v_1d_kms)

        label = rf"$t = {snap['t_Myr']:.1f}\ \mathrm{{Myr}}$"

        ax_rho.plot(r_kpc, rho_phys, color=color, label=label, lw=1.8)
        ax_v.plot(r_kpc, v_over_c, color=color, label=label, lw=1.8)
        ax_m.plot(r_kpc, m_phys, color=color, label=label, lw=1.8)
        ax_kn.plot(r_kpc, Knudsen, color=color, label=label, lw=1.8)

    # Subplot 1: Density
    ax_rho.set_ylabel(r"$\rho \quad [\mathrm{M}_{\odot} / \mathrm{kpc}^3]$")
    ax_rho.set_xscale("log")
    ax_rho.set_yscale("log")
    ax_rho.set_title(r"\textbf{Density vs. Radius}")
    ax_rho.grid(True, which="both", ls="--", alpha=0.5)
    ax_rho.legend(loc="lower left", frameon=True)

    # Subplot 2: Velocity / c
    ax_v.set_ylabel(r"$v / c$")
    ax_v.set_xscale("log")
    ax_v.set_yscale("log")
    ax_v.set_title(r"\textbf{Velocity Dispersion vs. Radius} ($v/c$)")
    ax_v.grid(True, which="both", ls="--", alpha=0.5)

    # Subplot 3: Enclosed Mass
    ax_m.set_xlabel(r"$r \quad [\mathrm{kpc}]$")
    ax_m.set_ylabel(r"$M_\chi \quad [\mathrm{M}_{\odot}]$")
    ax_m.set_xscale("log")
    ax_m.set_yscale("log")
    ax_m.set_title(r"\textbf{Enclosed Mass vs. Radius}")
    ax_m.grid(True, which="both", ls="--", alpha=0.5)

    # Subplot 4: Knudsen Number
    ax_kn.set_xlabel(r"$r \quad [\mathrm{kpc}]$")
    ax_kn.set_ylabel(r"$\mathrm{Kn} = \lambda / H$")
    ax_kn.set_xscale("log")
    ax_kn.set_yscale("log")
    ax_kn.axhline(1.0, color="gray", linestyle="-", lw=1.0)
    ax_kn.text(0.98, 1.3, r"LMFP", transform=ax_kn.get_yaxis_transform(),
               ha="right", va="bottom", color="gray")
    ax_kn.text(0.98, 0.75, r"SMFP", transform=ax_kn.get_yaxis_transform(),
               ha="right", va="top", color="gray")
    ax_kn.set_title(r"\textbf{Knudsen Number vs. Radius} ($\mathrm{Kn} = \lambda / H$)")
    ax_kn.grid(True, which="both", ls="--", alpha=0.5)

    plt.suptitle(r"\textbf{Profile Evolution for SIDM Halo}", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with PdfPages(save_path) as pdf:
            pdf.savefig(fig)
            plt.close(fig)
        print(f"PDF file successfully generated at:\n{save_path}")
    except PermissionError:
        alt_path = save_path.with_name(f"{save_path.stem}_new.pdf")
        with PdfPages(alt_path) as pdf:
            pdf.savefig(fig)
            plt.close(fig)
        print(
            f"WARNING: File {save_path} is currently open in another application.\n"
            f"Plot saved to alternative file:\n{alt_path}"
        )


if __name__ == "__main__":
    generate_plots(filepath, output_pdf_path, num_profiles=7)
