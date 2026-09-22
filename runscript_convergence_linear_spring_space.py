import logging
import sys

import numpy as np
import porepy as pp

sys.path.append("../")
import logging
import os
import sys

import numpy as np
import porepy as pp

from model_convergence_linear_spring import SpringTypeLinearConvergenceSetup

logger = logging.getLogger(__name__)

# Run-parameters
RUN_MODEL = False
COARSE = True

# Base directory (project root = where this script lives)
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Top-level results folder
MAIN_RESULTS_DIR = os.path.join(SCRIPT_PATH, "convergence_analysis_results")

# Figures
FIGURES_DIR = os.path.join(MAIN_RESULTS_DIR, "figures")

# Ensure directory exist
os.makedirs(FIGURES_DIR, exist_ok=True)

# File names and file paths
FILENAME_ROCK = f"errors.txt"
FILENAME_FRACTURE = f"errors_fracture.txt"
FIGURE_PATH = os.path.join(FIGURES_DIR, "convergence_linear_spring_model_space.png")
            
# Error file header
header = "num_cells, num_time_steps, displacement_error, traction_error\n"

for setup_type in ["shear", "compressive"]:
    SETUP_FOLDER = f"SL_{setup_type}_space"
    OUTPUT_DIR = os.path.join(MAIN_RESULTS_DIR, SETUP_FOLDER)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    RESULTS_FILE_PATH_ROCK = os.path.join(OUTPUT_DIR, FILENAME_ROCK)
    RESULTS_FILE_PATH_FRACTURE = os.path.join(OUTPUT_DIR, FILENAME_FRACTURE)

    # Set fracture stiffness, wave amplitude, maximum time step and maximum cell size
    A = 5.0e-5
    fracture_stiffness = 2.0e11
    final_time = 2.0e-5
    dt_max = final_time / 20
    cs_max = 1.0e-3

    if RUN_MODEL:
        refinement_coefficients = np.arange(0, 2) if COARSE else np.arange(0, 4)
        # refinement_coefficients = [1]
        for coeff in refinement_coefficients:
            if coeff == 0:
                with open(RESULTS_FILE_PATH_ROCK, "w") as file:
                    file.write(header)
                with open(RESULTS_FILE_PATH_FRACTURE, "w") as file:
                    file.write(header)
            dt = 6.25e-8

            time_manager = pp.TimeManager(
                schedule=[0.0, final_time],
                dt_init=dt,
                constant_dt=True,
            )

            lmbda_shale = 11.0e9
            mu_shale = 10.2e9
            rho_shale = 2400.0

            lmbda_generic_rock = 4.0e9
            mu_generic_rock = 4.0e9
            rho_generic_rock = 2600.0

            solid_vals = {
                "fracture_gap": 0.0,
                "dilation_angle": 0.0,
                "shear_modulus": mu_generic_rock,
                "lame_lambda": lmbda_generic_rock,
                "density": rho_generic_rock,
                "fracture_normal_stiffness": fracture_stiffness,
                "fracture_tangential_stiffness": fracture_stiffness,
            }

            solid = pp.SolidConstants(**solid_vals)

            solid_values_inner_region = {
                "mu_parallel": mu_shale,
                "mu_orthogonal": mu_shale,
                "lambda_parallel": 0.0,
                "lambda_orthogonal": 0.0,
                "volumetric_compr_lambda": lmbda_shale,
                "rho": rho_shale,
            }

            # Include the time_manager to the model params dictionary
            params = {
                "time_manager": time_manager,
                "meshing_arguments": {
                    "cell_size": cs_max / 2**coeff
                },
                "folder_name": f"visualization_spring_{setup_type}_ref_{coeff}",
                "grid_type": "simplex",
                "material_constants": {"solid": solid},
                "discontinuity_location": 25.0e-3,
                "wave_frequency": 100.0e3,
                "compressive_setup": setup_type == "compressive",
                "heterogeneity_type": "simple",
                "solid_values_inner_region": solid_values_inner_region,
                "solver_statistics_file_name": "solver_statistics.json",
                "wave_amplitude": A,
                "times_to_export": [final_time],
                "linear_solver": {
                    # "options": 
                    #     {"gmres": {
                    #         "ksp_monitor": None,
                    #     }},
                },
            }

            model = SpringTypeLinearConvergenceSetup(params)
            model.filename_path = RESULTS_FILE_PATH_ROCK
            model.filename_path_fracture = RESULTS_FILE_PATH_FRACTURE
            other_params = {
                "progressbars": True,
                "nl_max_iterations": 50,
                "nl_convergence_inc_atol": 1.0e-8,
                "nl_convergence_res_atol": 1.0e-8,
            }

            runner = pp.ModelRunner(model=model, params=other_params)
            runner.run()

# Plot convergence errors
import matplotlib.pyplot as plt
from plotting.plot_utils import draw_multiple_loglog_slopes

setups = ["compressive", "shear"]
ks = [(2.0e11, "2.0e11")]

all_data = {}

for setup_type in setups:
    OUTPUT_DIR = os.path.join(MAIN_RESULTS_DIR, f"SL_{setup_type}_space")

    for k_val, k_str in ks:
        filename_bulk = os.path.join(OUTPUT_DIR, "errors.txt")
        filename_frac = os.path.join(OUTPUT_DIR, "errors_fracture.txt")

        data_bulk = np.loadtxt(filename_bulk, delimiter=",", skiprows=1)
        data_frac = np.loadtxt(filename_frac, delimiter=",", skiprows=1)

        if data_bulk.ndim == 1:
            data_bulk = data_bulk.reshape(1, -1)
        if data_frac.ndim == 1:
            data_frac = data_frac.reshape(1, -1)

        all_data[f"{setup_type}_bulk"] = data_bulk
        all_data[f"{setup_type}_frac"] = data_frac

# Create figure with subplots (2x2: shear rock, shear frac, comp rock, comp frac)
fig, axes = plt.subplots(2, 2, figsize=(18, 10))

# Order: comp rock, comp frac, shear rock, shear frac
configurations = [
    ("compressive", "bulk", axes[0, 0], "Compressive setup: Rock"),
    ("compressive", "frac", axes[0, 1], "Compressive setup: Fracture"),
    ("shear", "bulk", axes[1, 0], "Shear setup: Rock"),
    ("shear", "frac", axes[1, 1], "Shear setup: Fracture"),
]

for setup_type, region, ax, title in configurations:
    data = all_data[f"{setup_type}_{region}"]

    num_cells = data[:, 0]
    num_time_steps = data[:, 1]
    displacement_error = data[:, 2]
    traction_error = data[:, 3]

    # Calculate x-axis: (N_x * N_t)^(1/3) for bulk, (N_x * N_t)^(1/2) for fracture
    exponent = 1.0 / 3.0 if region == "bulk" else 1.0 / 2.0
    x_vals = (num_cells) ** exponent

    # Plot displacement error: black with circle markers
    if region == "bulk":
        disp_label = "Displacement"
        trac_label = "Traction"
    else:
        disp_label = "Displacement Jump"
        trac_label = "Fracture Contact Traction"

    ax.loglog(
        x_vals,
        displacement_error,
        "o-",
        color="black",
        linewidth=3.5,
        markersize=15,
        label=disp_label,
    )

    # Plot traction error: dashed gray with diamond markers
    ax.loglog(
        x_vals,
        traction_error,
        "D--",
        color="darkgray",
        linewidth=3.5,
        markersize=10,
        label=trac_label,
    )

    ax.set_title(title, fontsize=22, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=22)

    # Increase tick label font size
    ax.tick_params(axis="both", which="major", labelsize=22)
    ax.tick_params(axis="both", which="minor", labelsize=0)

    # Add convergence slope triangles for displacement error (smaller)
    draw_multiple_loglog_slopes(
        fig,
        ax,
        origin=(1.1 * x_vals[-2], 1.25 * traction_error[-2]),
        triangle_width=0.8,
        slopes=[-2],
        dashed_extra_slopes=True,
        inverted=False,
        color="black",
        fontsize_factor=2.0,
    )

    # Y labels: show label only on left column; on right column remove label text
    # but keep the tick numbering. X labels: show only on bottom row; hide
    # x-axis numbering on the top row.
    if region == "bulk":  # Left column
        ax.set_ylabel(r"Relative error", fontsize=24)
    else:
        ax.set_ylabel("", fontsize=2)

    if setup_type == "shear":  # Bottom row
        xlabel = (
            r"$(N_x)^{1/3}$"
            if region == "bulk"
            else r"$(N_x)^{1/2}$"
        )
        ax.set_xlabel(xlabel, fontsize=24)
    else:
        # hide x-axis numbering for top row
        ax.tick_params(axis="x", which="both", labelbottom=False)


# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved convergence figure to {FIGURE_PATH}")
