import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ============================================================
# STYLE
# ============================================================
FONT_SIZE_2 = 21
FONT_SIZE_3 = 22

mpl.rcParams.update(
    {
        "contour.negative_linestyle": "solid",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": FONT_SIZE_3,
        "axes.titlesize": FONT_SIZE_3,
        "xtick.labelsize": FONT_SIZE_2,
        "ytick.labelsize": FONT_SIZE_2,
        "legend.fontsize": FONT_SIZE_2,
    }
)

# ============================================================
# PATHS
# ============================================================
root_dir = "/workdir/mpsa_newmark_fractures/convergence_analysis_results"
cases = ["C_Coul_Lin_space", "C_Coul_BB_space"]

fig_dir = os.path.join(root_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)


# ============================================================
# DATA LOADER (UNCHANGED)
# ============================================================
def load_data(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    arrays = []
    for block in content.split("],"):
        block = block.strip()
        if not block:
            continue

        block = block.replace("[", "").replace("]", "").replace("\n", " ")
        vals = [float(x) for x in block.split() if x.strip()]
        arrays.append(vals)

    return np.vstack(arrays).astype(float)


# ============================================================
# LOAD ALL (ONLY CHANGE: SLIP COMPUTATION)
# ============================================================
def load_all():
    data = {}

    # Tolerance for undefined normal traction. Chosen upon inspection of the traction
    # values.
    atol = 1.0e-9

    for case in cases:
        base = os.path.join(root_dir, case)

        # ========================================================
        # DISPLACEMENTS
        # ========================================================
        tangential = load_data(os.path.join(base, "displacement_jump_t.txt"))
        normal = load_data(os.path.join(base, "displacement_jump_n.txt"))

        # ========================================================
        # FRACTURE OPENING
        # ========================================================
        opening = load_data(os.path.join(base, "fracture_opening.txt"))

        # ========================================================
        # TRACTIONS IN PA
        # ========================================================
        tau_t = load_data(os.path.join(base, "traction_t.txt"))
        tau_n = load_data(os.path.join(base, "traction_n.txt"))

        # ========================================================
        # TRACTIONS NONDIM
        # ========================================================
        tau_t_nondim = load_data(os.path.join(base, "traction_t_nondim.txt"))
        tau_n_nondim = load_data(os.path.join(base, "traction_n_nondim.txt"))

        # ========================================================
        # SLIP TENDENCY
        # ========================================================
        slip = np.full_like(tau_t_nondim, np.nan)

        valid = ~np.isclose(tau_n_nondim, 0.0, atol=atol)
        slip[valid] = np.abs(tau_t_nondim[valid]) / np.abs(tau_n_nondim[valid])

        # ========================================================
        # STORE
        # ========================================================
        data[case] = {
            "tangential": tangential,
            "normal": normal,
            "opening": opening,
            "traction_t": tau_t,
            "traction_n": tau_n,
            "slip": slip,
        }

    return data

def plot_line_last_timestep(data, case="C_Coul_Lin"):
    base = data[case]

    # ========================================================
    # LAST TIMESTEP (same as reference script)
    # ========================================================
    opening = base["opening"][-1]
    slip = base["slip"][-1]
    tau_t = base["traction_t"][-1]
    tau_n = base["traction_n"][-1]

    # ========================================================
    # SAFE: use opening to find mesh size
    # ========================================================
    n = len(opening)

    opening = opening[:n]
    slip = slip[:n]
    tau_t = tau_t[:n]
    tau_n = tau_n[:n]

    fracture_length = 0.016
    x = np.linspace(0, fracture_length, n)[::-1]

    # ========================================================
    # PLOT
    # ========================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # ===== TOP (exact reference structure) =====
    ax1.set_ylabel("Fracture opening [m]", color="black")
    ax1.set_ylim(
        -np.nanmax(opening) * 0.02, np.nanmax(opening) * 1.3
    ) 
    ax1_right = ax1.twinx()
    ax1_right.set_ylim(-np.nanmax(slip) * 0.018, np.nanmax(slip) * 1.6)
    ax1_right.set_ylabel("Slip Tendency", color="black")

    line1 = ax1.plot(x, opening, "k-", linewidth=3.0, label="Fracture opening")
    line2 = ax1_right.plot(x, slip, "darkgray", linewidth=3.0, label="Slip tendency")

    ax1.tick_params(axis="x", labelbottom=False)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ===== BOTTOM =====
    ax2.set_xlabel("Position along fracture [m]")
    ax2.set_ylabel("Contact traction [Pa]")
    ax2.set_ylim(
        -1.0e7, 2.75e7
    ) 
    ax2.plot(x, tau_t, "g-", linewidth=3.0, label="Tangential traction")
    ax2.plot(x, tau_n, "k--", linewidth=3.0, label="Friction bound (positive)")
    ax2.plot(x, -tau_n, "k-.", linewidth=3.0, label="Friction bound (negative)")

    ax2.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    ax1.set_xlim(0, fracture_length)
    ax2.set_xlim(0, fracture_length)

    plt.tight_layout()

    out = os.path.join(fig_dir, f"line_plots_{case}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", out)


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    data = load_all()
    for case in cases:
        plot_line_last_timestep(data, case)
