import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# PATHS AND GRAPHICAL STYLE CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
# Usage: python dimensionless_units.py [results_dir]
# Default: Hernquist baseline results (this script plots the analytical Hernquist baryon model).
results_dir = (
    Path(sys.argv[1]).resolve()
    if len(sys.argv) > 1
    else (BASE_DIR / ".." / "01-baseline" / "hernquist" / "results").resolve()
)
plots_dir = (BASE_DIR / "output" / "hernquist_rho_vs_m").resolve()
plots_dir.mkdir(parents=True, exist_ok=True)

# Safe search for .txt result files
list_of_files = sorted(results_dir.glob("result_*.txt"))
if not list_of_files:
    raise FileNotFoundError(f"No result_*.txt files found in: {results_dir}")

print(f"Found {len(list_of_files)} file(s) to process.\n")

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "legend.fontsize": 8,
})

# ==============================================================================
# ANALYTICAL MODEL: HERNQUIST PROFILE (DIMENSIONLESS UNITS)
# ==============================================================================
def density_baryon(r, mass_norm, ars):
    r_safe = np.maximum(r, 1e-10)
    return (2.0 * mass_norm * ars) / (r_safe * (r_safe + ars) ** 3)


def mass_baryon(r, mass_norm, ars):
    r_safe = np.maximum(r, 0.0)
    return mass_norm * (r_safe / (r_safe + ars)) ** 2


def get_num_layers_from_file(filepath):
    """
    Extract the number of layers from the file header or filename.
    """
    filename = filepath.name
    match_name = re.search(r"(\d+)\s*layers", filename, re.IGNORECASE)
    if match_name:
        return match_name.group(1)

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(15):
                line = f.readline()
                if not line:
                    break
                match = re.search(r"layers?\s*[:=]\s*(\d+)", line, re.IGNORECASE)
                if match:
                    return match.group(1)
    except Exception:
        pass

    return "N/A"


def parse_simulation_profiles(filepath):
    """
    Read the simulation file line-by-line to prevent OSError and MemoryError.
    """
    profiles = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            line = f.readline()
            if not line:
                break

            line_str = line.strip()
            if not line_str or line_str.startswith((
                "Input", "Initial", "Number", "Total",
                "Save", "Abs", "Cross", "Conduction",
                "Baryon", "time,"
            )):
                continue

            parts = line_str.split()
            if len(parts) == 2:
                try:
                    t_val = float(parts[0])
                    tstep = int(parts[1])

                    # Read the 5 consecutive data arrays
                    R = np.fromstring(f.readline(), sep=" ")
                    Rho = np.fromstring(f.readline(), sep=" ")
                    M = np.fromstring(f.readline(), sep=" ")
                    _ = np.fromstring(f.readline(), sep=" ")  # u
                    _ = np.fromstring(f.readline(), sep=" ")  # L

                    profiles.append({
                        "time": t_val,
                        "step": tstep,
                        "R": R,
                        "Rho": Rho,
                        "M": M,
                    })
                except (ValueError, IndexError):
                    pass

    return profiles


# ==============================================================================
# LOOP OVER ALL SIMULATION FILES
# ==============================================================================
for file_idx, filepath in enumerate(list_of_files, 1):
    base_filename = filepath.name
    name_without_ext = filepath.stem
    print(f"[{file_idx}/{len(list_of_files)}] Processing: {base_filename}")

    num_layers = get_num_layers_from_file(filepath)
    profiles = parse_simulation_profiles(filepath)

    if not profiles:
        print(f"  -> WARNING: No valid profiles extracted from {base_filename}. Skipping.")
        continue

    # ==========================================================================
    # CREATE PLOT
    # ==========================================================================
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)

    num_profiles = len(profiles)

    # Equally spaced sampling by snapshot index
    num_target = min(12, num_profiles)
    selected_indices = np.linspace(0, num_profiles - 1, num_target, dtype=int).tolist()

    colors = plt.cm.coolwarm(np.linspace(0, 0.85, len(selected_indices)))

    # Plot simulated profiles
    for idx, color in zip(selected_indices, colors):
        p = profiles[idx]
        current_time = p["time"]

        label_text = rf"$t/t_0 = {current_time:.2f}$"
        ax.plot(p["M"], p["Rho"], color=color, lw=1.8, label=label_text)

    # ==========================================================================
    # DYNAMIC ASYMPTOTE: MASS AT STEEPEST SLOPE IN THE FINAL SNAPSHOT
    # ==========================================================================
    last_prof = profiles[selected_indices[-1]]
    valid_mask = (last_prof["M"] > 0) & (last_prof["Rho"] > 0)
    m_valid = last_prof["M"][valid_mask]
    rho_valid = last_prof["Rho"][valid_mask]

    log_M = np.log10(m_valid)
    log_Rho = np.log10(rho_valid)

    dlog_Rho_dlog_M = np.abs(np.gradient(log_Rho, log_M))
    max_slope_idx = np.argmax(dlog_Rho_dlog_M)

    M_asymptotic = m_valid[max_slope_idx]

    ax.axvline(x=M_asymptotic, color="black", linestyle=":", lw=1.2)

    if M_asymptotic < 1e-2:
        m_val, m_exp = f"{M_asymptotic:.2e}".split("e")
        val_str = rf"{m_val} \times 10^{{{int(m_exp)}}}"
        label_asymptote = rf"$M_{{\mathrm{{in}}}}/M_0 = {val_str}$"
    else:
        label_asymptote = rf"$M_{{\mathrm{{in}}}}/M_0 = {M_asymptotic:.3f}$"

    ax.text(
        M_asymptotic * 1.1,
        1e8,
        label_asymptote,
        rotation=90,
        fontsize=10,
        verticalalignment="bottom",
    )

    # Axes and grid formatting
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-5, 1.0)
    ax.set_ylim(1e-2, 1e15)
    ax.set_xlabel(r"$M_\chi / M_0$", fontsize=12)
    ax.set_ylabel(r"$\rho_\chi / \rho_s$", fontsize=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    # Legend
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, ncol=2)

    if num_layers != "N/A":
        ax.set_title(rf"\textbf{{Layers: {num_layers}}}", fontsize=14, pad=10)
    else:
        # Escape underscores for LaTeX rendering
        safe_title = base_filename.replace("_", r"\_")
        ax.set_title(rf"\textbf{{{safe_title}}}", fontsize=11, pad=10)

    # ==========================================================================
    # SAVE PDF
    # ==========================================================================
    if num_layers != "N/A" and f"{num_layers}layers" not in name_without_ext.lower():
        output_filename = f"{name_without_ext}_{num_layers}layers.pdf"
    else:
        output_filename = f"{name_without_ext}.pdf"

    output_path = plots_dir / output_filename
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"  -> PDF saved to: {output_path} (Asymptote found at M/M0 = {M_asymptotic:.4e})\n")

print("Processing and PDF generation completed!")
