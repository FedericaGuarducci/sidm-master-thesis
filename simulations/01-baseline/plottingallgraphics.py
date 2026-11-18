import glob
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# CONFIGURAZIONE PERCORSI E STILE
# ==============================================================================
results_dir = Path(r"C:\Users\feddy\Desktop\FedeFISICA\Gaggero\Programmi\9-CNNoBaryons\CN-nobarLayering\results8")
plots_dir = results_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# Ricerca sicura dei file .txt
list_of_files = [Path(p) for p in glob.glob(str(results_dir / "*.txt"))]
if not list_of_files:
    raise FileNotFoundError(f"Nessun file .txt trovato in: {results_dir}")

print(f"Trovati {len(list_of_files)} file da elaborare.\n")

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "legend.fontsize": 8,
})

# ==============================================================================
# MODELLO ANALITICO: PROFILO DI HERNQUIST (UNITA' ADIMENSIONALI)
# ==============================================================================
def density_baryon(r, mass_norm, ars):
    r_safe = np.maximum(r, 1e-10)
    return (2.0 * mass_norm * ars) / (r_safe * (r_safe + ars)**3)

def mass_baryon(r, mass_norm, ars):
    r_safe = np.maximum(r, 0.0)
    return mass_norm * (r_safe / (r_safe + ars))**2

def get_num_layers_from_file(filepath):
    """
    Estrae il numero di layers leggendo solo le prime righe del file
    o cercando nel nome del file.
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
    Legge il file di simulazione riga per riga per evitare OSError ed errori di memoria.
    """
    profiles = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            line = f.readline()
            if not line:
                break

            line_str = line.strip()
            if not line_str or line_str.startswith(("Input", "Initial", "Number", "Total",
                                                    "Save", "Abs", "Cross", "Conduction",
                                                    "Baryon", "time,")):
                continue

            parts = line_str.split()
            if len(parts) == 2:
                try:
                    t_val = float(parts[0])
                    tstep = int(parts[1])

                    # Legge i 5 array di dati consecutivi
                    R = np.fromstring(f.readline(), sep=" ")
                    Rho = np.fromstring(f.readline(), sep=" ")
                    M = np.fromstring(f.readline(), sep=" ")
                    u = np.fromstring(f.readline(), sep=" ")
                    L = np.fromstring(f.readline(), sep=" ")

                    profiles.append({
                        "time": t_val,
                        "step": tstep,
                        "R": R,
                        "Rho": Rho,
                        "M": M
                    })
                except (ValueError, IndexError):
                    pass

    return profiles

# ==============================================================================
# LOOP SU TUTTI I FILE
# ==============================================================================
for file_idx, filepath in enumerate(list_of_files, 1):
    base_filename = filepath.name
    name_without_ext = filepath.stem
    print(f"[{file_idx}/{len(list_of_files)}] Elaborazione: {base_filename}")

    num_layers = get_num_layers_from_file(filepath)
    profiles = parse_simulation_profiles(filepath)

    if not profiles:
        print(f"  -> ATTENZIONE: Nessun profilo valido estratto da {base_filename}. Salto il file.")
        continue

    # ==========================================================================
    # CREAZIONE DEL GRAFICO
    # ==========================================================================
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)

    num_profiles = len(profiles)

    # ==========================================================================
    # CAMPIONAMENTO PROFILI EQUISPAZIATI NEL TEMPO FISICO
    # ==========================================================================
# ==========================================================================
    # CAMPIONAMENTO PROFILI EQUISPAZIATI NEL NUMERO DI SNAPSHOT
    # ==========================================================================
    num_target = min(12, num_profiles)
    selected_indices = np.linspace(0, num_profiles - 1, num_target, dtype=int).tolist()

    colors = plt.cm.coolwarm(np.linspace(0, 0.85, len(selected_indices)))

    # Parametri profilo (Normalizzazioni)
    my_mass_norm = 0.1
    my_scale_norm = 0.1

    # Griglia per il calcolo analitico
    r = np.logspace(-5.0, 1.0, 500)


    # Tracciamento profili simulati
    for idx, color in zip(selected_indices, colors):
        p = profiles[idx]
        current_time = p["time"]

        label_text = rf"$t/t_0 = {current_time:.2f}$"
        ax.plot(p["M"], p["Rho"], color=color, lw=1.8, label=label_text)

    # ==========================================================================
    # ASINTOTO DINAMICO: MASSA ALLA PENDENZA MASSIMA NELL'ULTIMO SNAPSHOT
    # ==========================================================================
    last_prof = profiles[selected_indices[-1]]
    log_M = np.log10(last_prof["M"])
    log_Rho = np.log10(last_prof["Rho"])

    dlog_Rho_dlog_M = np.abs(np.gradient(log_Rho, log_M))
    max_slope_idx = np.argmax(dlog_Rho_dlog_M)

    M_asymptotic = last_prof["M"][max_slope_idx]

    ax.axvline(x=M_asymptotic, color="black", linestyle=":", lw=1.2)

    if M_asymptotic < 1e-2:
        val_str = f"{M_asymptotic:.2e}".replace("e-0", r" \times 10^{-").replace("e-", r" \times 10^{-") + "}"
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

    # Formattazione assi e griglia
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-5, 1.0)
    ax.set_ylim(1e-2, 1e15)
    ax.set_xlabel(r"$M_\chi / M_0$", fontsize=12)
    ax.set_ylabel(r"$\rho_\chi / \rho_s$", fontsize=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    # Legenda
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, ncol=2)

    if num_layers != "N/A":
        ax.set_title(rf"\textbf{{Layers: {num_layers}}}", fontsize=14, pad=10)
    else:
        ax.set_title(rf"\textbf{{{base_filename}}}", fontsize=11, pad=10)

    # ==========================================================================
    # SALVATAGGIO PDF
    # ==========================================================================
    if num_layers != "N/A" and f"{num_layers}layers" not in name_without_ext.lower():
        output_filename = f"{name_without_ext}_{num_layers}layers.pdf"
    else:
        output_filename = f"{name_without_ext}.pdf"

    output_path = plots_dir / output_filename
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"  -> PDF salvato in: {output_path} (Asintoto trovato a M/M0 = {M_asymptotic:.4e})\n")

print("Elaborazione e generazione PDF completata!")