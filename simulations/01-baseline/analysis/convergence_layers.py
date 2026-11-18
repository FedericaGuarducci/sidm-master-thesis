from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np

# Enable native LaTeX rendering engine (requires a local TeX installation)
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

# Target threshold for central density: 10^17
RHO_TARGET = 1e17


def extract_target_profile_exact(filepath, target_rho=RHO_TARGET, t0=1.0):
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    initial_layers = None
    for line in lines:
        if line.startswith("Number of layers:"):
            val_part = line.split(":")[1].strip()
            match = re.search(r"\d+", val_part)
            if match:
                initial_layers = int(match.group(0))
            break

    if initial_layers is None:
        match = re.search(r"(\d+)\s*layers", filepath.name, re.IGNORECASE)
        initial_layers = int(match.group(1)) if match else None

    if initial_layers is None:
        return None, None, None, None, None, None

    blocks = []
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) == 2:
            try:
                t_val = float(parts[0])
                _ = int(parts[1])  # step

                if i + 5 < len(lines):
                    raw_density = [float(x) for x in lines[i + 2].split()]
                    raw_mass = [float(x) for x in lines[i + 3].split()]

                    if raw_density and raw_mass and len(raw_density) == len(raw_mass):
                        blocks.append({
                            "time": t_val,
                            "rho_center": raw_density[0],
                            "density": np.array(raw_density),
                            "mass": np.array(raw_mass),
                        })
                    i += 6
                    continue
            except ValueError:
                pass
        i += 1

    if not blocks:
        return None, None, None, None, None, None

    idx_after = None
    for idx, b in enumerate(blocks):
        if b["rho_center"] >= target_rho:
            idx_after = idx
            break

    if idx_after is None:
        print(
            f"[INFO] {filepath.name}: Max central density = {blocks[-1]['rho_center']:.2e} (< {target_rho:.1e})"
        )
        return None, None, None, None, None, None

    if idx_after == 0:
        b = blocks[0]
        final_time = b["time"]
        density_interp = b["density"]
        mass_interp = b["mass"]
    else:
        b_before = blocks[idx_after - 1]
        b_after = blocks[idx_after]

        rho_c1 = b_before["rho_center"]
        rho_c2 = b_after["rho_center"]

        if rho_c1 > 0 and rho_c2 > 0:
            weight = (np.log10(target_rho) - np.log10(rho_c1)) / (
                np.log10(rho_c2) - np.log10(rho_c1)
            )
        else:
            weight = 0.5

        weight = np.clip(weight, 0.0, 1.0)
        final_time = b_before["time"] + weight * (b_after["time"] - b_before["time"])

        min_len = min(len(b_before["density"]), len(b_after["density"]))

        log_rho1 = np.log10(np.maximum(b_before["density"][:min_len], 1e-30))
        log_rho2 = np.log10(np.maximum(b_after["density"][:min_len], 1e-30))
        log_m1 = np.log10(np.maximum(b_before["mass"][:min_len], 1e-30))
        log_m2 = np.log10(np.maximum(b_after["mass"][:min_len], 1e-30))

        log_rho_interp = log_rho1 + weight * (log_rho2 - log_rho1)
        log_m_interp = log_m1 + weight * (log_m2 - log_m1)

        density_interp = 10**log_rho_interp
        mass_interp = 10**log_m_interp

        density_interp[0] = target_rho

    valid = (mass_interp > 0) & (density_interp > 0)
    mass_clean = mass_interp[valid]
    density_clean = density_interp[valid]

    t_ratio = final_time / t0

    asymptote_mass = None
    if len(mass_clean) > 2:
        log_m = np.log10(mass_clean)
        log_rho = np.log10(density_clean)
        slopes = np.diff(log_rho) / np.diff(log_m)
        min_slope_idx = np.argmin(slopes)
        asymptote_mass = mass_clean[min_slope_idx]

    return initial_layers, mass_clean, density_clean, t_ratio, final_time, asymptote_mass


def plot_exact_target_profiles(results_dir, file_glob="result_*.txt", save_dir=None, t0=1.0):
    results_dir = Path(results_dir)
    filenames = sorted(results_dir.glob(file_glob))

    if not filenames:
        print(f"[WARNING] No files found in {results_dir} matching pattern: {file_glob}")
        return

    plt.figure(figsize=(12, 7))

    # Accepted target layer counts (including 180)
    target_layers = {180, 200, 300, 400, 500, 600, 700, 800, 900, 1000}
    evolution_times = {}
    asymptotic_masses = {}

    plot_elements = []

    for filepath in filenames:
        try:
            num_layers, mass, density, t_ratio, total_time, asympt_val = extract_target_profile_exact(
                filepath, target_rho=RHO_TARGET, t0=t0
            )

            if num_layers is None or num_layers not in target_layers:
                continue

            evolution_times[num_layers] = total_time
            asymptotic_masses[num_layers] = asympt_val
            t_ratio_str = f"{t_ratio:.2f}"

            if asympt_val is not None:
                m_val, m_exp = f"{asympt_val:.2e}".split("e")
                m_exp = int(m_exp)
                asympt_str = rf"{m_val} \times 10^{{{m_exp}}}"

                label_text = (
                    rf"${num_layers}\mathrm{{\ layers\ }} "
                    rf"(t/t_0 = {t_ratio_str},\ M_{{\mathrm{{in}}}}/M_0 = {asympt_str})$"
                )
            else:
                label_text = rf"${num_layers}\mathrm{{\ layers\ }} (t/t_0 = {t_ratio_str})$"

            (line,) = plt.loglog(mass, density, label=label_text, linewidth=1.8)

            if asympt_val is not None:
                plt.axvline(
                    x=asympt_val,
                    color=line.get_color(),
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1.2,
                )

            plot_elements.append((num_layers, line, label_text))

            print(
                f"[PROCESSED] {filepath.name} | Layers: {num_layers} | t_target: {total_time:.2f} | M_in/M_0: {asympt_val}"
            )
        except Exception as e:
            print(f"[ERROR] Failed to process {filepath}: {e}")

    if not plot_elements:
        print("[WARNING] No profile matched the target density threshold or target layer numbers.")
        return

    # Sort plotted elements by layer count before building legend and ratio box
    plot_elements.sort(key=lambda x: x[0])

    box_lines = [r"$\mathrm{Relative\ Ratios}:$" + "\n"]
    sorted_present_layers = sorted(evolution_times.keys())
    pairs = list(zip(sorted_present_layers[1:], sorted_present_layers[:-1]))

    for n2, n1 in pairs:
        has_t = (n2 in evolution_times) and (n1 in evolution_times)
        has_m = (
            (n2 in asymptotic_masses)
            and (n1 in asymptotic_masses)
            and (asymptotic_masses[n2] is not None)
            and (asymptotic_masses[n1] is not None)
        )

        if has_t and has_m:
            r_t = evolution_times[n2] / evolution_times[n1]
            r_m = asymptotic_masses[n2] / asymptotic_masses[n1]
            box_lines.append(
                rf"$\bullet\ t_{{{n2}}}/t_{{{n1}}} = {r_t:.2f} \quad | \quad M_{{{n2}}}/M_{{{n1}}} = {r_m:.2f}$"
            )
        elif has_t:
            r_t = evolution_times[n2] / evolution_times[n1]
            box_lines.append(
                rf"$\bullet\ t_{{{n2}}}/t_{{{n1}}} = {r_t:.2f} \quad | \quad M_{{{n2}}}/M_{{{n1}}}:\ \mathrm{{N/A}}$"
            )

    if len(box_lines) > 1:
        box_text = "\n".join(box_lines)
        props = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.85)
        plt.gca().text(
            0.03,
            0.05,
            box_text,
            transform=plt.gca().transAxes,
            fontsize=8.5,
            verticalalignment="bottom",
            bbox=props,
        )

    plt.xlabel(r"$\mathrm{Enclosed\ Mass}\ M / M_0$", fontsize=12)
    plt.ylabel(r"$\mathrm{Density\ Profile}\ \rho / \rho_s$", fontsize=12)
    plt.title(
        r"$\mathrm{Dark\ Matter\ Density\ Profiles\ at\ Exact\ Central\ Threshold}\ \rho_{\mathrm{center}} = 10^{17}$",
        fontsize=13,
    )
    plt.grid(True, which="both", linestyle=":", alpha=0.5)

    sorted_handles = [item[1] for item in plot_elements]
    sorted_labels = [item[2] for item in plot_elements]

    plt.legend(
        sorted_handles,
        sorted_labels,
        title=r"$\mathrm{Number\ of\ Layers}$",
        fontsize=8.5,
        loc="upper right",
    )
    plt.tight_layout()

    output_dir = Path(save_dir) if save_dir is not None else Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "dm_layers_exact_10e17.pdf"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"\n[SUCCESS] PDF figure successfully saved to: {pdf_path}")

    plt.show()


if __name__ == "__main__":
    # Resolve paths relative to this script's directory (e.g., inside analysis/)
    BASE_DIR = Path(__file__).resolve().parent
    target_results_dir = (BASE_DIR / ".." / "spl" / "results").resolve()
    output_plots_dir = (BASE_DIR / "output").resolve()

    plot_exact_target_profiles(
        results_dir=target_results_dir,
        file_glob="result_*.txt",
        save_dir=output_plots_dir,
        t0=1.0,
    )
