from pathlib import Path

import meshio
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

base_dir = Path(".")

# Choose "time" or "space"
ANALYSIS_TYPES = ["time", "space"]

for ANALYSIS_TYPE in ANALYSIS_TYPES:
    # Maximum refinement levels for each analysis type
    max_levels = {
        "time": 5,
        "space": 4,
    }

    if ANALYSIS_TYPE not in max_levels:
        raise ValueError(
            f"Unknown analysis type '{ANALYSIS_TYPE}'. Choose 'time' or 'space'."
        )

    file_pattern = (
        "friction_self_convergence_ref_{level}_C_Coul_{model}_"
        f"{ANALYSIS_TYPE}/data_1_000000.vtu"
    )

    root_dir = "/workdir/mpsa_newmark_fractures/convergence_analysis_results"
    fig_dir = os.path.join(root_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    output_file = f"tangential_traction_Lin_vs_BB_{ANALYSIS_TYPE}.png"

    # Lighter shades for coarser refinements, darker for finer ones
    num_levels = max_levels[ANALYSIS_TYPE] + 1
    lin_colors = plt.cm.Blues(np.linspace(0.10, 0.90, num_levels))
    bb_colors = plt.cm.Reds(np.linspace(0.10, 0.90, num_levels))


    def read_tangential_traction(vtu_file):
        mesh = meshio.read(vtu_file)

        if "contact_traction_in_Pa" not in mesh.cell_data:
            raise KeyError(
                f"'contact_traction_in_Pa' not found in {vtu_file}"
            )

        x_blocks = []
        tangential_blocks = []

        for cells, traction in zip(
            mesh.cells, mesh.cell_data["contact_traction_in_Pa"]
        ):
            if len(cells.data) == 0:
                continue

            L = 0.016  # Fracture length [m]
            n = len(cells.data)  # Number of cells

            # X-coordinate of each cell center
            cell_x = (np.arange(n) + 0.5) * L / n

            # Reverse order if needed
            cell_x = cell_x[::-1]
            
            # Component 0 is the tangential traction
            tangential = traction[:, 0]

            x_blocks.append(cell_x)
            tangential_blocks.append(tangential)

        if not x_blocks:
            raise ValueError(f"No cell data found in {vtu_file}")

        x = np.concatenate(x_blocks)
        tangential = np.concatenate(tangential_blocks)

        order = np.argsort(x)
        return x[order], tangential[order]


    # Find available refinement levels across both models
    available_levels = []

    for level in range(max_levels[ANALYSIS_TYPE] + 1):
        for model in ("Lin", "BB"):
            vtu_file = base_dir / file_pattern.format(
                level=level,
                model=model,
            )
            if vtu_file.is_file():
                available_levels.append(level)
                break

    available_levels = sorted(set(available_levels))

    if not available_levels:
        raise FileNotFoundError(
            f"No VTU files found for analysis type '{ANALYSIS_TYPE}' "
            f"in {base_dir.resolve()}"
        )

    # Font sizes, Times-style font, and line widths
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 17,
        "legend.title_fontsize": 16,
        "lines.linewidth": 1.8,
        "axes.formatter.use_mathtext": True,
    })

    fig, axes = plt.subplots(
        2, 1,
        figsize=(10, 8),
        sharex=True,
        constrained_layout=True,
    )

    model_settings = [
        ("Lin", axes[0], lin_colors),
        ("BB", axes[1], bb_colors),
    ]

    finest_level = max(available_levels)

    for model, ax, colors in model_settings:
        for level in available_levels:
            vtu_file = base_dir / file_pattern.format(
                level=level,
                model=model,
            )

            if not vtu_file.is_file():
                print(f"WARNING: File not found, skipping: {vtu_file}")
                continue

            try:
                x, tangential = read_tangential_traction(vtu_file)
            except (KeyError, ValueError, OSError) as error:
                print(f"WARNING: Could not read {vtu_file}: {error}")
                continue

            # Finest available refinement is solid; others are dashed
            linestyle = "-" if level == finest_level else "--"

            # Cell boundaries for piecewise constant traction
            L = 0.016
            n = len(x)
            cell_edges = np.linspace(0, L, n + 1)

            ax.stairs(
                tangential,
                cell_edges,
                color=colors[level],
                linewidth=1.8,
                baseline=None,
                linestyle=linestyle,
                label=rf"$r = {level}$",
            )

        ax.set_ylabel("Traction [Pa]")
        ax.grid(True, alpha=0.25)

        # Integer tick values with the power of ten shown separately
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        formatter.set_useOffset(False)
        ax.yaxis.set_major_formatter(formatter)

        ax.tick_params(axis="both", which="major", labelsize=18)

        ax.legend(
            title=r"Tangential contact traction for:",
            ncol=3,
            frameon=True,
        )

    axes[1].set_xlabel("Position along fracture [m]")

    # Save the figure
    output_path = os.path.join(fig_dir, output_file)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output_path)