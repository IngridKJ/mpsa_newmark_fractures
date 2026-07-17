from model_example_compare_models import (
    CBB,
    CL,
    SL,
    SBB,
)
import porepy as pp
import os

COEFF = 3
# Base directory (project root = where this script lives)
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Top-level results folder
MAIN_RESULTS_DIR = os.path.join(
    SCRIPT_PATH, "simulation_example_results_compare_models"
)

# Ensure directories exist
os.makedirs(MAIN_RESULTS_DIR, exist_ok=True)

# Configurations and create directories for each one:
MODEL_TAGS = {
    "C_Coul_Lin": CL,
    "C_Coul_BB": CBB,
    "S_Lin_Lin": SL,
    "S_Lin_BB": SBB,
}

for tag in MODEL_TAGS:
    os.makedirs(os.path.join(MAIN_RESULTS_DIR, f"{tag}"), exist_ok=True)

# Model parameters
MATERIAL_CONSTANTS = {
    "fracture_gap": 0.0,
    "dilation_angle": 0.0,
    "friction_coefficient": 1.0,
    "shear_modulus": 4.0e9,
    "lame_lambda": 4.0e9,
    "density": 2600.0,
    "maximum_elastic_fracture_opening": 1.0e-5,
}

NUMERICAL_CONSTANTS = {
    "open_state_tolerance": 1.0e-12,
}

SOLVER_PARAMS = {
    "progressbars": True,
    "nl_max_iterations": 30,
    "nl_convergence_inc_atol": 1.0e-10,
    "nl_convergence_res_atol": 1.0e-10,
}


def make_parameter_dictionary(model_tag: str) -> dict:
    """Create model and solver parameters."""
    final_time = 1.0390625e-05
    num_steps = 190
    dt = final_time / num_steps

    time_manager = pp.TimeManager(
        schedule=[0.0, final_time], dt_init=dt, constant_dt=True
    )
    solid = pp.SolidConstants(**MATERIAL_CONSTANTS)
    numerical = pp.NumericalConstants(**NUMERICAL_CONSTANTS)
    params = {
        "time_manager": time_manager,
        "folder_name": f"simulation_example_{model_tag}",
        "model_tag": model_tag,
        "grid_type": "simplex",
        "meshing_arguments": {"cell_size": 16.0e-4 / (2**COEFF)},
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


def run_model(model_tag: str, model_class) -> None:
    params, run_params = make_parameter_dictionary(model_tag)
    model = model_class(params=params)
    model.results_dir = os.path.join(MAIN_RESULTS_DIR, f"{model_tag}")
    for filename in [
        "displacement_jump_n.txt",
        "displacement_jump_t.txt",
        "traction_n.txt",
        "traction_t.txt",
    ]:
        FILEPATH = os.path.join(model.results_dir, filename)
        if os.path.exists(FILEPATH):
            os.remove(FILEPATH)

    pp.ModelRunner(model, run_params).run()

if __name__ == "__main__":
    run_model("C_Coul_Lin", CL)
    run_model("C_Coul_BB", CBB)
    run_model("S_Lin_Lin", SL)
    run_model("S_Lin_BB", SBB)
