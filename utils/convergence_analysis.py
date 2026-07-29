"""Generic utilities for convergence analysis and plotting."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import os

import porepy as pp


class DataExtractor:
    """Extract data from mixed-dimensional models."""

    @staticmethod
    def matrix_grid(model) -> pp.Grid:
        """Get matrix grid (dim=2)."""
        return model.mdg.subdomains(dim=2)[0]

    @staticmethod
    def fracture_grid(model) -> pp.Grid:
        """Get fracture grid (dim=1)."""
        return model.mdg.subdomains(dim=1)[0]

    @staticmethod
    def extract_matrix_displacement(model) -> np.ndarray:
        """Extract matrix displacement."""
        sd = DataExtractor.matrix_grid(model)
        u = model.displacement([sd]).value(model.equation_system)
        u = np.asarray(u)

        if u.shape == (model.nd, sd.num_cells):
            return u
        if u.size == model.nd * sd.num_cells:
            return u.reshape((model.nd, sd.num_cells), order="F")
        raise ValueError(f"Could not interpret displacement shape {u.shape}.")

    @staticmethod
    def extract_fracture_displacement_jump(model) -> np.ndarray:
        """Extract fracture displacement jump."""
        frac_sd = model.mdg.subdomains(dim=model.nd - 1)
        u_jump = model.displacement_jump(frac_sd).value(model.equation_system)
        return np.asarray(u_jump).reshape((model.nd, -1), order="F")

    @staticmethod
    def extract_contact_traction(model) -> np.ndarray:
        """Extract contact traction."""
        frac_sd = model.mdg.subdomains(dim=model.nd - 1)
        traction = model.contact_traction(frac_sd).value(model.equation_system)
        return np.asarray(traction).reshape((model.nd, -1), order="F")


def compute_errors(
    coarse_model,
    fine_model,
    matrix_overlap_func,
    fracture_overlap_func,
    averaging_weight="volume",
    tol=1e-10,
):
    """Compute relative L2 errors between models.

    Parameters:
        coarse_model: Coarse model instance.
        fine_model: Fine model instance.
        matrix_overlap_func: Function to compute matrix overlap L2 components.
        fracture_overlap_func: Function to compute fracture overlap L2 components.
        averaging_weight: Weight for averaging ("volume" or similar).
        tol: Tolerance for overlap computation.

    Returns:
        Tuple of (matrix_error, fracture_displacement_error, traction_error).

    """
    ext = DataExtractor

    # Matrix displacement error
    matrix_num, matrix_den = matrix_overlap_func(
        sd_H=ext.matrix_grid(coarse_model),
        sd_h=ext.matrix_grid(fine_model),
        u_H=ext.extract_matrix_displacement(coarse_model),
        u_h=ext.extract_matrix_displacement(fine_model),
        weight=averaging_weight,
        tol=tol,
    )

    # Fracture displacement jump error
    frac_num, frac_den = fracture_overlap_func(
        sd_H=ext.fracture_grid(coarse_model),
        sd_h=ext.fracture_grid(fine_model),
        wH_cc=ext.extract_fracture_displacement_jump(coarse_model),
        wh_cc=ext.extract_fracture_displacement_jump(fine_model),
        weight=averaging_weight,
    )

    # Contact traction error
    trac_num, trac_den = fracture_overlap_func(
        sd_H=ext.fracture_grid(coarse_model),
        sd_h=ext.fracture_grid(fine_model),
        wH_cc=ext.extract_contact_traction(coarse_model),
        wh_cc=ext.extract_contact_traction(fine_model),
        weight=averaging_weight,
    )

    return (
        float(np.sqrt(matrix_num / matrix_den)),
        float(np.sqrt(frac_num / frac_den)),
        float(np.sqrt(trac_num / trac_den)),
    )


def save_errors(
    results_dir, model_tag, num_cells_matrix, num_cells_frac, num_time_steps, errors
):
    """Save error data to CSV files.

    Parameters:
        results_dir: Results directory path.
        model_tag: Model tag (e.g., "C_Coul_Lin", "C_Coul_BB").
        num_cells_matrix: Array of matrix cell counts.
        num_cells_frac: Array of fracture cell counts.
        num_time_steps: Array of time step counts.
        errors: Dict with keys "disp", "jump", "trac" containing error arrays.

    """
    error_dir = os.path.join(results_dir, f"{model_tag}")
    os.makedirs(error_dir, exist_ok=True)

    with open(os.path.join(error_dir, "errors.txt"), "w") as f:
        f.write("num_cells, num_time_steps, displacement_error\n")
        for nc, nt, err in zip(num_cells_matrix, num_time_steps, errors["disp"]):
            f.write(f"{nc}, {nt}, {err}\n")

    with open(os.path.join(error_dir, "errors_fracture.txt"), "w") as f:
        f.write(
            "num_cells, num_time_steps, "
            "displacement_error_fracture, traction_error_fracture\n"
        )
        for nc, nt, jump, trac in zip(
            num_cells_frac, num_time_steps, errors["jump"], errors["trac"]
        ):
            f.write(f"{nc}, {nt}, {jump}, {trac}\n")


def plot_convergence(
    results_dir,
    model_tag,
    x_matrix,
    x_fracture,
    errors_disp,
    errors_jump,
    errors_trac,
    draw_slopes_func=None,
):
    """Create convergence plot.

    Parameters:
        results_dir: Results directory path for saving figures.
        model_tag: Model tag (e.g., "C_Coul_Lin", "C_Coul_BB").
        x_matrix: x-axis values for matrix displacement.
        x_fracture: x-axis values for fracture quantities.
        errors_disp: Matrix displacement error array.
        errors_jump: Fracture displacement jump error array.
        errors_trac: Contact traction error array.
        draw_slopes_func: Optional function to draw convergence slopes.

    """
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.0,
            "mathtext.fontset": "dejavusans",
        }
    )

    plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Times"]
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.5))
    # Matrix displacement
    ax1.loglog(
        x_matrix, errors_disp, "o-", color="black", linewidth=3.0, label="Displacement"
    )
    ax1.set_xlabel(r"$(N_x\cdot N_t)^{1/3}$")
    ax1.set_ylabel("Relative error")

    if draw_slopes_func is not None:
        draw_slopes_func(
            fig,
            ax1,
            origin=(0.8 * x_matrix[-1], 1.1 * errors_disp[-1]),
            triangle_width=0.5,
            slopes=[-1.5],
            dashed_extra_slopes=True,
            inverted=True,
            color="black",
        )

    # Fracture quantities
    ax2.loglog(
        x_fracture,
        errors_jump,
        "s-",
        color="black",
        linewidth=3.0,
        label="Displacement jump",
    )
    ax2.loglog(
        x_fracture, errors_trac, "D-", color="dimgray", linewidth=3.0, label="Traction"
    )
    ax2.set_xlabel(r"$(N_x\cdot N_t)^{1/2}$")
    ax2.set_ylabel("Relative error")

    if draw_slopes_func is not None:
        draw_slopes_func(
            fig,
            ax2,
            origin=(0.8 * x_fracture[-1], 1.1 * errors_jump[-1]),
            triangle_width=0.5,
            slopes=[-1.5],
            dashed_extra_slopes=True,
            inverted=True,
            color="black",
        )

    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right",)

    plt.tight_layout()
    figures_dir = os.path.join(results_dir, "figures")
    figure_path = os.path.join(figures_dir, f"self_convergence_{model_tag}.png")
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {figure_path}")
    plt.close(fig)

def load_errors(results_dir, model_tag):
    """Load saved convergence data from disk."""
    error_dir = os.path.join(results_dir, model_tag)

    errors_disp, errors_jump, errors_trac = [], [], []
    num_cells_matrix, num_cells_frac, num_time_steps = [], [], []

    # read matrix displacement file
    with open(os.path.join(error_dir, "errors.txt"), "r") as f:
        next(f)  # header
        for line in f:
            nc, nt, err = line.strip().split(",")
            num_cells_matrix.append(int(nc))
            num_time_steps.append(int(nt))
            errors_disp.append(float(err))

    # read fracture file
    with open(os.path.join(error_dir, "errors_fracture.txt"), "r") as f:
        next(f)  # header
        for line in f:
            nc, nt, jump, trac = line.strip().split(",")
            num_cells_frac.append(int(nc))
            errors_jump.append(float(jump))
            errors_trac.append(float(trac))

    return (
        np.asarray(num_cells_matrix),
        np.asarray(num_cells_frac),
        np.asarray(num_time_steps),
        {
            "disp": np.asarray(errors_disp),
            "jump": np.asarray(errors_jump),
            "trac": np.asarray(errors_trac),
        },
    )