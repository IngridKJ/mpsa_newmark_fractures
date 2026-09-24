from __future__ import annotations

import porepy as pp
import numpy as np
import os
from model_convergence_contact_mechanics import (
    SelfConvergenceCBB,
    SelfConvergenceCL,
)
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import matplotlib.pyplot as plt
import matplotlib as mpl
from convergence_metrics import (
    matrix_overlap_l2_components_p1,
    fracture_overlap_l2_components_p1,
)
from plotting.plot_utils import draw_multiple_loglog_slopes

from utils.convergence_analysis import (
    DataExtractor,
    compute_errors,
    save_errors,
    load_errors,
)

# Run-parameters
RUN_MODELS = False
COARSE = False

# Refinement coefficients for the convergence analysis
COEFFS = [0, 1, 2] if COARSE else [0, 1, 2, 3, 4]

# Base directory (project root = where this script lives)
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Top-level results folder
MAIN_RESULTS_DIR = os.path.join(SCRIPT_PATH, "convergence_analysis_results")

# Figures
FIGURES_DIR = os.path.join(MAIN_RESULTS_DIR, "figures")

# Ensure directories exist
os.makedirs(MAIN_RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Configurations and create directories for each one:
MODEL_TAGS = {
    "C_Coul_Lin_space": SelfConvergenceCL,
    "C_Coul_BB_space": SelfConvergenceCBB,
}

for tag in MODEL_TAGS:
    os.makedirs(os.path.join(MAIN_RESULTS_DIR, f"{tag}"), exist_ok=True)

# File names and file paths
FILENAME_ROCK = "errors.txt"
FILENAME_FRACTURE = "errors_fracture.txt"


# Model parameters
MATERIAL_CONSTANTS = {
    "fracture_gap": 0.0,
    "dilation_angle": 0.0,
    "friction_coefficient": 1.0,
    "shear_modulus": 4.0e9,
    "lame_lambda": 4.0e9,
    "density": 2600.0,
    "maximum_elastic_fracture_opening": 5.0e-5,
}

NUMERICAL_CONSTANTS = {
    "open_state_tolerance": 1.0e-12,
}

SOLVER_PARAMS = {
    "progressbars": True,
    "nl_max_iterations": 30,
    "nl_convergence_inc_atol": 1.0e-8,
    "nl_convergence_res_atol": 1.0e-8,
}


def make_parameter_dictionary(
    coeff: float, model_tag: str, reference_flag: bool
) -> dict:
    """Create model and solver parameters."""
    final_time = 1.0e-5 * (140 / 160)
    num_steps = 40 * 2**COEFFS[-1]
    dt = final_time / num_steps

    time_manager = pp.TimeManager(
        schedule=[0.0, final_time], dt_init=dt, constant_dt=True
    )
    solid = pp.SolidConstants(**MATERIAL_CONSTANTS)
    numerical = pp.NumericalConstants(**NUMERICAL_CONSTANTS)
    params = {
        "time_manager": time_manager,
        "folder_name": f"friction_self_convergence_ref_{coeff}_{model_tag}",
        "model_tag": model_tag,
        "reference_flag": reference_flag,
        "grid_type": "simplex",
        "meshing_arguments": {
            "cell_size": 16.0e-4 / (2**coeff)
        },
        "material_constants": {"solid": solid, "numerical": numerical},
        "wave_amplitude": 5.0e-5,
        "wave_frequency": 100.0e3,
        "solver_statistics_file_name": "solver_statistics.json",
        "linear_solver": {
            # "options":
            #     {"gmres": {
            #         "ksp_monitor": None,
            #     }},
        },
    }
    return params, SOLVER_PARAMS


def run_model(
    coeff: float, model_class, model_tag: str, reference_flag: bool = False
) -> pp.Model:
    """Run a single model."""
    params, run_params = make_parameter_dictionary(coeff, model_tag, reference_flag)

    # Clear old output files for this run
    if reference_flag:
        MODEL_RESULTS_DIR = os.path.join(MAIN_RESULTS_DIR, model_tag)
        for filename in [
            "displacement_jump_n.txt",
            "displacement_jump_t.txt",
            "traction_n.txt",
            "traction_t.txt",
            "traction_n_nondim.txt",
            "traction_t_nondim.txt",
            "fracture_opening.txt",
            "fracture_cc.txt",
            "slip_tendency.txt",
            "errors.txt",
            "errors_fracture.txt",
        ]:
            FILEPATH = os.path.join(MODEL_RESULTS_DIR, filename)
            if os.path.exists(FILEPATH):
                os.remove(FILEPATH)
    model = model_class(params=params)

    run_params = {
            "progressbars": True,
            "nl_convergence_criteria": {
                "inc_rel": pp.IncrementBasedRelativeCriterion(
                    tol=1e-8, metric=pp.VariableBasedEuclideanMetric(model)
                ),
            },
            "nl_divergence_criteria": {
                "max_iter": pp.MaxIterationsCriterion(max_iterations=25),
                "inc_nan": pp.IncrementBasedNanCriterion(),
                "res_nan": pp.ResidualBasedNanCriterion(),
            },
        }
    pp.ModelRunner(model, run_params).run()
    return model


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
    ax1.set_xlabel(r"$(N_x)^{1/2}$")
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
    ax2.set_xlabel(r"$N_x$")
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

def run_convergence_analysis(
    model_tag: str, model_class, run_models: bool = True
) -> None:
    """Run self-convergence analysis for a model and/or plot the results."""
    if run_models:
        print(
            f"\n{'=' * 70}\nRunning {model_tag} time convergence analysis\n{'=' * 70}\n"
        )
        fine_model = run_model(COEFFS[-1], model_class, model_tag, reference_flag=True)

        errors, cells, time_steps, cells_times = (
            {"disp": [], "jump": [], "trac": []},
            {"matrix": [], "fracture": []},
            [],
            {"matrix": [], "fracture": []},
        )

        for coeff in COEFFS[:-1]:
            print(f"Running coarse model h = {coeff}")
            coarse_model = run_model(coeff, model_class, model_tag)

            err_disp, err_jump, err_trac = compute_errors(
                coarse_model,
                fine_model,
                matrix_overlap_l2_components_p1,
                fracture_overlap_l2_components_p1,
            )
            errors["disp"].append(err_disp)
            errors["jump"].append(err_jump)
            errors["trac"].append(err_trac)

            ext = DataExtractor
            sd_matrix = ext.matrix_grid(coarse_model)
            sd_frac = ext.fracture_grid(coarse_model)
            ts = coarse_model.time_manager.time_index

            cells["matrix"].append(sd_matrix.num_cells)
            cells["fracture"].append(sd_frac.num_cells)
            time_steps.append(ts)
            cells_times["matrix"].append(sd_matrix.num_cells)
            cells_times["fracture"].append(sd_frac.num_cells)

            print(
                f"  Matrix: {sd_matrix.num_cells} cells, "
                f"Fracture: {sd_frac.num_cells} cells, "
                f"Time steps: {ts}"
            )
            print(
                f"  Errors - Disp: {err_disp:.3e}, "
                f"Jump: {err_jump:.3e}, "
                f"Trac: {err_trac:.3e}"
            )

        # Convert to arrays
        for key in errors:
            errors[key] = np.asarray(errors[key])
        for key in cells:
            cells[key] = np.asarray(cells[key])
        time_steps = np.asarray(time_steps)

        save_errors(
            MAIN_RESULTS_DIR,
            model_tag,
            cells["matrix"],
            cells["fracture"],
            time_steps,
            errors,
        )
        print(
            f"\nError files saved to {os.path.join(MAIN_RESULTS_DIR, f'{model_tag}')}"
        )
    else:
        print("Loading saved convergence data (no simulation run)...")

        cells_matrix, cells_frac, time_steps, errors = load_errors(
            MAIN_RESULTS_DIR, model_tag
        )

        cells_times = {
            "matrix": cells_matrix,
            "fracture": cells_frac,
        }

    x_matrix = np.asarray(cells_times["matrix"])
    x_fracture = np.asarray(cells_times["fracture"])
    plot_convergence(
        MAIN_RESULTS_DIR,
        model_tag,
        x_matrix,
        x_fracture,
        errors["disp"],
        errors["jump"],
        errors["trac"],
        draw_multiple_loglog_slopes,
    )


if __name__ == "__main__":
    run_convergence_analysis(
        "C_Coul_Lin_space", SelfConvergenceCL, run_models=RUN_MODELS
    )
    run_convergence_analysis(
        "C_Coul_BB_space", SelfConvergenceCBB, run_models=RUN_MODELS
    )
