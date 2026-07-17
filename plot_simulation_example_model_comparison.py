import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ============================================================
# STYLE
# ============================================================
FONT_SIZE_2 = 23
FONT_SIZE_3 = 23

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
root_dir = (
    "/workspaces/momentum_balance_inertia/simulation_example_results_compare_models"
)

cases_SL_SBB_CL_CBB = ["S_Lin_Lin", "S_Lin_BB", "C_Coul_Lin", "C_Coul_BB"]

all_cases = [
    cases_SL_SBB_CL_CBB,
]

fig_dir = os.path.join(root_dir, "figures/heatmaps")
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
# LOAD ALL
# ============================================================
def load_all():
    data = {}
    for case in cases:
        base = os.path.join(root_dir, case)

        # ========================================================
        # DISPLACEMENTS
        # ========================================================
        tangential = load_data(os.path.join(base, "displacement_jump_t.txt"))
        normal = load_data(os.path.join(base, "displacement_jump_n.txt"))

        # ========================================================
        # TRACTIONS IN PA
        # ========================================================
        tau_t = load_data(os.path.join(base, "traction_t.txt"))
        tau_n = load_data(os.path.join(base, "traction_n.txt"))

        # ========================================================
        # STORE
        # ========================================================
        data[case] = {
            "tangential": tangential,
            "normal": normal,
            "traction_t": tau_t,
            "traction_n": tau_n,
        }

    return data


# ============================================================
# AXES
# ============================================================
def build_axes(shape):
    n_time, n_space = shape

    final_time = 1.0390625e-05
    time = np.linspace(0, final_time, n_time)

    L = 0.016
    space = np.linspace(0, L, n_space)[::-1]

    return time, space


# ============================================================
# LIMITS
# ============================================================
def compute_limits(data, key):
    all_vals = np.concatenate([entry[key].ravel() for entry in data.values()])
    return np.nanmin(all_vals), np.nanmax(all_vals)


# ============================================================
# CONTOURS
# ============================================================
N_CONTOURS = {
    "tangential": 10,
    "normal": 10,
}


def make_levels(vmin, vmax, n_levels):
    levels = np.linspace(vmin, vmax, n_levels)
    if vmin < 0 < vmax:
        levels = np.unique(np.sort(np.append(levels, 0.0)))
    return levels


# ============================================================
# PLOT
# ============================================================
def plot_all(data):
    limits = {
        "tangential": compute_limits(data, "tangential"),
        "normal": compute_limits(data, "normal"),
    }

    contour_levels = {
        key: make_levels(*limits[key], N_CONTOURS[key]) for key in N_CONTOURS
    }

    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad("white")
    cmap_pw = mpl.colormaps["RdBu_r"].copy()

    t_cut = 0.2e-5

    fields = [
        ("Tangential jump", "tangential", r"$[\![u]\!]_\tau$", True),
        ("Normal jump", "normal", r"$[\![u]\!]_n$", True),
    ]

    for title, key, cbar_label, use_contours in fields:
        fig, subfigs = plt.subplots(
            1, len(cases), figsize=(20, 6), layout="constrained"
        )

        ims = []

        for i, case in enumerate(cases):
            base = data[case]
            t_axis, x_axis = build_axes(base[key].shape)

            vmin, vmax = limits[key]
            ax = subfigs[i]
            if key in ("normal", "tangential"):
                vmax_abs = max(abs(vmin), abs(vmax))
                vmin = -vmax_abs
                vmax = vmax_abs

                im = ax.pcolormesh(
                    x_axis,
                    t_axis,
                    base[key],
                    cmap=cmap_pw,
                    vmin=vmin,
                    vmax=vmax,
                    shading="auto",
                )
            else:
                im = ax.pcolormesh(
                    x_axis,
                    t_axis,
                    base[key],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    shading="auto",
                )
            ims.append(im)

            ax.set_xlabel("Position along fracture [m]")
            ax.set_ylabel("Time [s]")
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

            if use_contours:
                t_start = np.searchsorted(t_axis, t_cut)

                masked = np.full_like(base[key], np.nan)
                masked[t_start:, :] = base[key][t_start:, :]

                levels = contour_levels[key]

                neg_levels = levels[levels < 0]
                pos_levels = levels[levels > 0]

                if len(neg_levels) > 0:
                    ax.contour(
                        x_axis,
                        t_axis,
                        masked,
                        levels=neg_levels,
                        colors="paleturquoise",
                        linewidths=2.5,
                        alpha=0.5,
                    )

                if len(pos_levels) > 0:
                    ax.contour(
                        x_axis,
                        t_axis,
                        masked,
                        levels=pos_levels,
                        colors="lightcoral",
                        linewidths=2.5,
                        alpha=0.5,
                    )

                if 0.0 in levels:
                    ax.contour(
                        x_axis,
                        t_axis,
                        masked,
                        levels=[0.0],
                        colors="black",
                        linewidths=2.5,
                    )

        cbar = fig.colorbar(ims[-1], ax=subfigs, location="right", shrink=0.9, pad=0.01)
        cbar.set_label(cbar_label, fontsize=FONT_SIZE_3)

        out = os.path.join(fig_dir, f"{key}_{str(cases)}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

        print("Saved:", out)

# ============================================================
# TRACTION RATIO |tau_t / tau_n| FOR COULOMB MODELS
# ============================================================
def plot_traction_ratio(data):
    coulomb_cases = [case for case in cases_SL_SBB_CL_CBB if case.startswith("C")]

    fig, subfigs = plt.subplots(
        1,
        len(coulomb_cases) + 2,
        figsize=(20, 6),
        layout="constrained",
    )

    # Keep subfigures 1 and 2 completely empty
    subfigs[0].axis("off")
    subfigs[1].axis("off")

    if len(coulomb_cases) == 1:
        subfigs = [subfigs]

    ratios = {}

    # --------------------------------------------------------
    # Compute absolute ratios safely
    # --------------------------------------------------------
    atol = 1.0e-9

    for case in coulomb_cases:
        tau_t = data[case]["traction_t"]
        tau_n = data[case]["traction_n"]

        ratio = np.full_like(tau_t, np.nan)

        valid = ~np.isclose(tau_n, 0.0, atol=atol)
        ratio[valid] = np.abs(tau_t[valid] / tau_n[valid])

        ratios[case] = ratio

    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad("white")

    ims = []

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    for i, case in enumerate(coulomb_cases):

        ratio = ratios[case]

        t_axis, x_axis = build_axes(ratio.shape)

        ax = subfigs[i + 2]

        im = ax.pcolormesh(
            x_axis,
            t_axis,
            ratio,
            cmap=cmap,
            shading="auto",
        )

        ims.append(im)

        # ax.set_title(case)
        ax.set_xlabel("Position along fracture [m]")
        ax.set_ylabel("Time [s]")
        ax.xaxis.set_major_formatter(
            mticker.FormatStrFormatter("%.3f")
        )

    cbar = fig.colorbar(
        ims[-1],
        ax=subfigs,
        location="right",
        shrink=0.9,
        pad=0.045,
    )

    cbar.set_label(
        r"$s$",
        fontsize=FONT_SIZE_3 + 4,
    )

    out = os.path.join(
        fig_dir,
        "slip_tendency_model_comparison.png",
    )

    plt.savefig(
        out,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("Saved:", out)

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    for cases in all_cases:
        data = load_all()
        plot_all(data)

    plot_traction_ratio(data)