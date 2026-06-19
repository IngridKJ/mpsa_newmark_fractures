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

for setup_type in ["shear", "compressive"]:
    # Prepare path for generated output files
    folder_name_parent = "convergence_analysis_results/"
    folder_name_results = "SL_" + setup_type
    folder_name = folder_name_parent + folder_name_results

    header = "num_cells, num_time_steps, displacement_error, traction_error\n"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Set fracture stiffness, maximum time step and maximum cell size
    ks = [
        (2.0e11, "2.0e11"),
    ]
    final_time = 2.0e-5
    dt_max = final_time / 20
    cs_max = 1.0e-3

    for k in ks:
        filename = f"errors.txt"
        filename = os.path.join(output_dir, filename)
        filename_fracture = f"errors_fracture.txt"
        filename_fracture = os.path.join(output_dir, filename_fracture)

        refinements = np.arange(0, 4)
        for refinement_coefficient in refinements:
            if refinement_coefficient == 0:
                with open(filename, "w") as file:
                    file.write(header)
                with open(filename_fracture, "w") as file:
                    file.write(header)
            dt = dt_max / 2**refinement_coefficient

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

            fracture_stiffness = k[0]

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
            folder_name = f"spring_{setup_type}_refinement_{refinement_coefficient}"
            params = {
                "time_manager": time_manager,
                "meshing_arguments": {"cell_size": cs_max / 2**refinement_coefficient},
                "folder_name": folder_name,
                "grid_type": "simplex",
                "material_constants": {"solid": solid},
                "discontinuity_location": 25.0e-3,
                "wave_frequency": 100.0e3,
                "compressive_setup": setup_type == "compressive",
                "heterogeneity_type": "simple",
                "solid_values_inner_region": solid_values_inner_region,
                "wave_amplitude": 5.0e-5,
                "times_to_export": [final_time],
            }

            model = SpringTypeLinearConvergenceSetup(params)
            model.filename_path = filename
            model.filename_path_fracture = filename_fracture
            other_params = {"progressbars": True, "max_iterations": 50}

            runner = pp.ModelRunner(model=model, params=other_params)
            runner.run()

# Plot convergence errors
import matplotlib.pyplot as plt
from plotting.plot_utils import draw_multiple_loglog_slopes

# Set font to DejaVu Serif (serif font available on Linux)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Times"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# Create figures directory
script_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(script_dir, "convergence_analysis_results", "figures")
os.makedirs(figures_dir, exist_ok=True)

# Setup configurations
setups = ["compressive", "shear"]
ks = [(2.0e11, "2.0e11")]

# Prepare data for all configurations
all_data = {}

for setup_type in setups:
    folder_name_parent = "convergence_analysis_results/"
    folder_name = folder_name_parent + folder_name_results
    output_dir = os.path.join(script_dir, folder_name)

    for k_val, k_str in ks:
        # Read bulk error data
        filename_bulk = os.path.join(output_dir, f"errors.txt")
        data_bulk = np.loadtxt(filename_bulk, delimiter=",", skiprows=1)

        # Read fracture error data
        filename_frac = os.path.join(output_dir, f"errors_fracture.txt")
        data_frac = np.loadtxt(filename_frac, delimiter=",", skiprows=1)

        if data_bulk.ndim == 1:
            data_bulk = data_bulk.reshape(1, -1)
        if data_frac.ndim == 1:
            data_frac = data_frac.reshape(1, -1)

        all_data[f"{setup_type}_bulk"] = data_bulk
        all_data[f"{setup_type}_frac"] = data_frac

# Create figure with subplots (2x2: shear rock, shear frac, comp rock, comp frac)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

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
    x_vals = (num_cells * num_time_steps) ** exponent

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
    ax.legend(fontsize=16)

    # Increase tick label font size
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.tick_params(axis="both", which="minor", labelsize=18)

    # Add convergence slope triangles for displacement error (smaller)
    draw_multiple_loglog_slopes(
        fig,
        ax,
        origin=(0.8 * x_vals[-1], 1.2 * traction_error[-1]),
        triangle_width=0.6,
        slopes=[-2],
        dashed_extra_slopes=True,
        inverted=True,
        color="black",
        fontsize_factor=1.8,
    )

    # Y labels: show label only on left column; on right column remove label text
    # but keep the tick numbering. X labels: show only on bottom row; hide
    # x-axis numbering on the top row.
    if region == "bulk":  # Left column
        ax.set_ylabel(r"Relative error", fontsize=22)
    else:
        ax.set_ylabel("", fontsize=2)

    if setup_type == "shear":  # Bottom row
        xlabel = (
            r"$(N_x \cdot N_t)^{1/3}$"
            if region == "bulk"
            else r"$(N_x \cdot N_t)^{1/2}$"
        )
        ax.set_xlabel(xlabel, fontsize=22)
    else:
        # hide x-axis numbering for top row
        ax.tick_params(axis="x", which="both", labelbottom=False)


# Adjust layout
plt.tight_layout()

# Save figure
figure_path = os.path.join(figures_dir, "convergence_linear_spring_model.png")
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved convergence figure to {figure_path}")
