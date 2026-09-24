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

from models_nonlinear_fracture_deformation import SpringTypeBartonBandisConvergenceSetup
logger = logging.getLogger(__name__)

# Run-parameters
RUN_MODEL = False
COARSE = False

# Base directory (project root = where this script lives)
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Top-level results folder
MAIN_RESULTS_DIR = os.path.join(SCRIPT_PATH, "convergence_analysis_results")

# Subfolders
RESULTS_DIR = os.path.join(MAIN_RESULTS_DIR, "S_Lin_BB_space")
FIGURES_DIR = os.path.join(MAIN_RESULTS_DIR, "figures")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Files
RESULTS_FILE_PATH = os.path.join(RESULTS_DIR, "errors.txt")
FIGURE_PATH = os.path.join(FIGURES_DIR, "convergence_nonlinear_spring_model_space.png")

# Header for result files
header = "num_cells, num_time_steps, transmission_coefficient_error\n"

# Set final time, amount of time-steps and the time-step size
final_time = 2.0e-5
dt_max = final_time / 40
cs_max = 1.0e-3
As_and_kns = [
    (5.0e-8, 2.0e11),
]
M = 100000
u_max = 1.0e-7
if RUN_MODEL:
    for A, kn in As_and_kns:
        refinements = np.arange(0, 2) if COARSE else np.arange(0, 4)
        dt = dt_max / 2**refinements[-1]
        for refinement_coefficient in refinements:
            dx = cs_max / 2**refinement_coefficient

            if refinement_coefficient == 0:
                with open(RESULTS_FILE_PATH, "w") as file:
                    file.write(header)

            time_manager = pp.TimeManager(
                schedule=[0.0, final_time],
                dt_init=dt,
                constant_dt=True,
            )

            wave_frequency = 100.0e3
            mu = 4.0e9
            lmbda = 4.0e9
            rho = 2600.0

            time_manager = pp.TimeManager(
                schedule=[0.0, final_time],
                dt_init=dt,
                constant_dt=True,
            )
            solid_vals = {
                "fracture_gap": 0.0,
                "dilation_angle": 0.0,
                "shear_modulus": mu,
                "lame_lambda": lmbda,
                "density": rho,
                "fracture_normal_stiffness": kn,
                "fracture_tangential_stiffness": kn,
                "maximum_elastic_fracture_opening": u_max,
            }
            solid = pp.SolidConstants(**solid_vals)

            # Include the time_manager to the model params dictionary
            params = {
                "time_manager": time_manager,
                "folder_name": f"visualization_BB_ref_{refinement_coefficient}",
                "meshing_arguments": {"cell_size": dx},
                "grid_type": "simplex",
                "material_constants": {"solid": solid},
                "discontinuity_location": 25.0e-3,
                "wave_amplitude": A,
                "steps_per_period": M,
                "wave_frequency": wave_frequency,
                "linear_solver": {
                    # "options": 
                    #     {"gmres": {
                    #         "ksp_monitor": None,
                    #     }},
                },
            }

            model = SpringTypeBartonBandisConvergenceSetup(params)
            model.filename_path = RESULTS_FILE_PATH
            other_params = {"progressbars": True, "max_iterations": 50}
            runner = pp.ModelRunner(model, other_params)
            runner.run()

# Plot convergence errors
import matplotlib.pyplot as plt
from plotting.plot_utils import draw_multiple_loglog_slopes

# Set font to DejaVu Serif (serif font available on Linux)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Times"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# Read error data
data = np.loadtxt(RESULTS_FILE_PATH, delimiter=",", skiprows=1)

if data.ndim == 1:
    # Only one data point, reshape to 2D
    data = data.reshape(1, -1)

num_cells = data[:, 0]
num_time_steps = data[:, 1]
x_axis = (num_cells) ** (1 / 2)
transmission_coeff_error = data[:, 2]

# Create figure with error vs mesh refinement
fig, ax = plt.subplots(figsize=(12, 6))
ax.loglog(
    x_axis,
    transmission_coeff_error,
    "o-",
    color="k",
    linewidth=4,
    markersize=15,
    label="Transmission coefficient",
)

ax.set_xlabel("$N_x^{1/2}$", fontsize=24)
ax.set_ylabel("Relative error", fontsize=24)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=22)

# Increase tick label font size
ax.tick_params(axis="both", which="major", labelsize=24)
ax.tick_params(axis="both", which="minor", labelsize=20)

# Add convergence slope triangles
draw_multiple_loglog_slopes(
    fig,
    ax,
    origin=(0.7 * x_axis[-1], 2.25 * transmission_coeff_error[-1]),
    triangle_width=1.0,
    slopes=[-2, -1],  # 1st and 2nd order convergence
    dashed_extra_slopes=True,
    inverted=False,
    color="black",
    fontsize_factor=2.2,
)

# Save figure
plt.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved figure to {FIGURE_PATH}")
