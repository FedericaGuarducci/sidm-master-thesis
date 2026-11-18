from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

# ==============================================================================
# GRAPHICAL RENDERING CONFIGURATION (LaTeX Look & Feel without external deps)
# ==============================================================================
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"  # Use native LaTeX Computer Modern font

# ==============================================================================
# PATHS AND PHYSICAL PARAMETERS CONFIGURATION
# ==============================================================================
# Resolve path relative to this script's directory (e.g., inside analysis/)
BASE_DIR = Path(__file__).resolve().parent
target_filepath = (
    BASE_DIR / ".." / "spl" / "results" / "result_20261007-180layers.txt"
).resolve()

# Scale radius r_s 
R_S = 3.2

if not target_filepath.exists():
    raise FileNotFoundError(f"Could not find target file: {target_filepath}")

print(f"[INFO] Target file selected: {target_filepath.name}")


def find_inflection_radius(m_arr, rho_arr, r_norm_arr):
    """1. Identify the inflection mass M_inflect from log10(rho) vs log10(M).
    2. Interpolate the corresponding dimensionless inflection radius (r/r_s)_inflect.
    """
    if len(m_arr) == 0 or len(rho_arr) == 0 or len(r_norm_arr) == 0:
        return None, None
    if not (len(m_arr) == len(rho_arr) == len(r_norm_arr)):
        min_len = min(len(m_arr), len(rho_arr), len(r_norm_arr))
        m_arr, rho_arr, r_norm_arr = (
            m_arr[:min_len],
            rho_arr[:min_len],
            r_norm_arr[:min_len],
        )

    valid_mask = (r_norm_arr > 1e-6) & (m_arr > 1e-6) & (rho_arr > 0)

    m_v = m_arr[valid_mask]
    rho_v = rho_arr[valid_mask]
    r_v = r_norm_arr[valid_mask]

    if len(m_v) < 10:
        return None, None

    sort_idx = np.argsort(m_v)
    m_s, rho_s, r_s_arr = m_v[sort_idx], rho_v[sort_idx], r_v[sort_idx]

    u_idx = np.concatenate(([True], np.diff(m_s) > 0))
    m_s, rho_s, r_s_arr = m_s[u_idx], rho_s[u_idx], r_s_arr[u_idx]

    log_m = np.log10(m_s)
    log_rho = np.log10(rho_s)
    log_r = np.log10(r_s_arr)

    m_inflect = None

    try:
        spl = UnivariateSpline(log_m, log_rho, k=4, s=0.08 * len(log_m))
        d2_spl = spl.derivative(n=2)
        d1_spl = spl.derivative(n=1)

        roots = d2_spl.roots()
        margin = 0.05 * (log_m[-1] - log_m[0])
        valid_roots = [
            r for r in roots if (log_m[0] + margin) < r < (log_m[-1] - margin)
        ]

        if valid_roots:
            best_log_m = max(valid_roots, key=lambda r: abs(d1_spl(r)))
            m_inflect = 10**best_log_m
    except Exception:
        pass

    if m_inflect is None:
        d1 = np.gradient(log_rho, log_m)
        d2 = np.gradient(d1, log_m)
        if len(d2) > 10:
            zero_crossings = np.where(np.diff(np.sign(d2[5:-5])))[0] + 5
            if len(zero_crossings) > 0:
                best_idx = zero_crossings[np.argmax(np.abs(d1[zero_crossings]))]
                m_inflect = m_s[best_idx]

    if m_inflect is None:
        return None, None

    log_r_inflect = np.interp(np.log10(m_inflect), log_m, log_r)
    r_inflect_norm = 10**log_r_inflect

    return r_inflect_norm, m_inflect


# ==============================================================================
# PHASE 1 & 2: FULL FILE SCAN AND SELECTION OF EXACTLY 8 SNAPSHOTS
# ==============================================================================
block_offsets = []

with open(target_filepath, "r", encoding="utf-8", errors="ignore") as f:
    while True:
        pos = f.tell()
        line = f.readline()
        if not line:
            break
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                _ = float(parts[0])
                _ = int(parts[1])
                block_offsets.append(pos)
                for _ in range(5):
                    f.readline()
            except ValueError:
                continue

total_snapshots = len(block_offsets)
if total_snapshots == 0:
    raise RuntimeError(f"No valid data blocks found in {target_filepath.name}")

print(f"[INFO] Total snapshots in file: {total_snapshots}")
print("[INFO] Scanning for snapshots with layer changes...")

all_layer_changes = []
prev_n_layers = None

with open(target_filepath, "r", encoding="utf-8", errors="ignore") as f:
    for idx, offset in enumerate(block_offsets):
        f.seek(offset)

        header = f.readline().strip().split()
        line_r = f.readline()
        f.readline()  # Rho
        f.readline()  # M
        f.readline()  # u
        f.readline()  # L

        r_arr_raw = np.fromstring(line_r, dtype=float, sep=" ")
        n_layers = len(r_arr_raw)

        if prev_n_layers is None or n_layers != prev_n_layers:
            all_layer_changes.append((idx, n_layers, prev_n_layers))
            prev_n_layers = n_layers

last_snap_idx = total_snapshots - 1
if not any(item[0] == last_snap_idx for item in all_layer_changes):
    with open(target_filepath, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(block_offsets[last_snap_idx])
        _ = f.readline()
        line_r = f.readline()
        last_n_layers = len(np.fromstring(line_r, dtype=float, sep=" "))
    all_layer_changes.append((last_snap_idx, last_n_layers, None))

TARGET_COUNT = 8

if len(all_layer_changes) <= TARGET_COUNT:
    selected_changes = all_layer_changes
else:
    first_item = all_layer_changes[0]
    last_item = all_layer_changes[-1]
    middle_items = all_layer_changes[1:-1]

    sampled_indices = np.linspace(
        0, len(middle_items) - 1, TARGET_COUNT - 2, dtype=int
    )
    selected_changes = (
        [first_item] + [middle_items[p] for p in sampled_indices] + [last_item]
    )

parsed_snapshots = []
with open(target_filepath, "r", encoding="utf-8", errors="ignore") as f:
    for snap_idx, curr_n, prev_n in selected_changes:
        offset = block_offsets[snap_idx]
        f.seek(offset)

        header = f.readline().strip().split()
        t_val = float(header[0])
        step_val = int(header[1])

        line_r = f.readline()
        line_rho = f.readline()
        line_m = f.readline()
        f.readline()  # u
        f.readline()  # L

        r_arr_raw = np.fromstring(line_r, dtype=float, sep=" ")
        r_arr_norm = r_arr_raw / R_S

        rho_arr = np.fromstring(line_rho, dtype=float, sep=" ")
        m_arr = np.fromstring(line_m, dtype=float, sep=" ")

        r_inflect_norm, m_inflect = find_inflection_radius(
            m_arr, rho_arr, r_arr_norm
        )

        parsed_snapshots.append({
            "time": t_val,
            "step": step_val,
            "R_norm": r_arr_norm,
            "M": m_arr,
            "Rho": rho_arr,
            "r_inflect_norm": r_inflect_norm,
            "m_inflect": m_inflect,
            "n_layers": curr_n,
            "prev_n_layers": prev_n,
            "is_final": (snap_idx == last_snap_idx),
        })

print(f"[SUCCESS] Selected {len(parsed_snapshots)} snapshots for plotting:")
for snap in parsed_snapshots:
    tag = " [FINAL SNAPSHOT]" if snap["is_final"] else ""
    print(
        f"  - Step {snap['step']:6d} | t = {snap['time']:6.2f} | N_layers = {snap['n_layers']:3d}{tag}"
    )
print()

# ==============================================================================
# PHASE 3: PLOT GENERATION (2x4 GRID)
# ==============================================================================
r_all = [d["R_norm"] for d in parsed_snapshots]
rmin = min([np.min(r) for r in r_all if np.min(r) > 0])
rmax = max([np.max(r) for r in r_all])
bins = np.logspace(np.log10(rmin), np.log10(rmax), 71)

max_y = 0
for r_data in r_all:
    counts, _ = np.histogram(r_data, bins=bins)
    max_y = max(max_y, max(counts))

n_rows, n_cols = 2, 4
fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(4.2 * n_cols, 4.2 * n_rows), sharex=True, sharey=True
)
axes = axes.flatten()

colors = cm.plasma(np.linspace(0, 0.85, len(parsed_snapshots)))

for i, (data, color) in enumerate(zip(parsed_snapshots, colors)):
    ax = axes[i]
    r_data = data["R_norm"]
    current_step = data["step"]
    current_time = data["time"]
    r_inflect_norm = data["r_inflect_norm"]
    m_inflect = data["m_inflect"]
    n_layers = data["n_layers"]

    ax.hist(
        r_data,
        bins=bins,
        histtype="stepfilled",
        alpha=0.6,
        color=color,
        edgecolor=color,
    )

    if r_inflect_norm is not None:
        r_val, r_exp = f"{r_inflect_norm:.3e}".split("e")
        r_exp = int(r_exp)
        r_str_formatted = f"{r_val} \\times 10^{{{r_exp}}}"

        ax.axvline(
            x=r_inflect_norm,
            color="black",
            linestyle="--",
            lw=1.5,
            label=f"$r_{{inflect}}/r_s = {r_str_formatted}$",
        )
        subtitle_text = (
            f"$r/r_s = {r_str_formatted}$\n$(M = {m_inflect:.4f})$"
        )
    else:
        subtitle_text = "Inflection not detected"

    ax.set_xscale("log")
    ax.set_ylim(0, max_y + 2)

    final_tag = " (Final)" if data["is_final"] else ""
    main_title = f"Step {current_step}{final_tag} ($t = {current_time:.2f}$) [$N_{{layers}} = {n_layers}$]"

    ax.set_title(f"{main_title}\n{subtitle_text}", fontsize=8.5)
    ax.grid(True, which="both", ls="--", alpha=0.4)

# Hide unused subplots if fewer than 8 snapshots were selected
for j in range(len(parsed_snapshots), len(axes)):
    axes[j].set_visible(False)

fig.supxlabel(r"$r / r_s$", fontsize=12)
fig.supylabel("Layer Count per Bin", fontsize=12)
fig.suptitle(
    f"Selected {len(parsed_snapshots)} Snapshots of Layer Redistribution (Including Final)",
    fontsize=13,
    y=0.98,
)

fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94], h_pad=2.0, w_pad=1.5)
plt.show()
