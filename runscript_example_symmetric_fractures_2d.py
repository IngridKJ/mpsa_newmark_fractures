import numpy as np
import porepy as pp

from models_nonlinear_fracture_deformation import (
    ContactModelBartonBandisGapFunction,
)

# Define a units object to handle unit conversions. Here we use meters as base unit,
# hence the default length unit is "m" and the conversion will have no practical effect.
# Nevertheless, it is good practice to use the units framework to ensure consistency.
units = pp.Units()

# Run-parameters
SAVE_FIGURES = True
COARSE = True

class GeometryBoundaryConditionAndWaveFunction:
    def set_fractures(self) -> None:
        """Setting fractures."""
        frac_1_points = self.units.convert_units(
            np.array([[1, 4], [5.5, 7]]) * 25.0e-3 / 8, "m"
        )

        frac_2_points = self.units.convert_units(
            np.array([[2, 3], [6.5, 5.0]]) * 25.0e-3 / 8, "m"
        )

        frac_3_points = self.units.convert_units(
            np.array([[6, 7], [7, 6]]) * 25.0e-3 / 8, "m"
        )

        frac_4_points = self.units.convert_units(
            np.array([[1, 4], [2.5, 1]]) * 25.0e-3 / 8, "m"
        )

        frac_5_points = self.units.convert_units(
            np.array([[2, 3], [1.5, 3]]) * 25.0e-3 / 8, "m"
        )
        frac_6_points = self.units.convert_units(
            np.array([[6, 7], [1, 2]]) * 25.0e-3 / 8, "m"
        )

        # Create the line fractures
        frac_1 = pp.LineFracture(frac_1_points)
        frac_2 = pp.LineFracture(frac_2_points)
        frac_3 = pp.LineFracture(frac_3_points)
        frac_4 = pp.LineFracture(frac_4_points)
        frac_5 = pp.LineFracture(frac_5_points)
        frac_6 = pp.LineFracture(frac_6_points)

        self._fractures = [
            frac_1,
            frac_2,
            frac_3,
            frac_4,
            frac_5,
            frac_6,
        ]

    def set_domain(self) -> None:
        """Domain of the problem."""
        x = self.units.convert_units(25.0e-3, "m")
        y = self.units.convert_units(25.0e-3, "m")
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x, "ymin": 0, "ymax": y}
        self._domain = pp.Domain(box)

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Method for assigning boundary condition type.

        Parameters:
            sd: The subdomain whose bc type is assigned.

        Return:
            The boundary condition object.

        """
        # Fetch boundary sides and assign type of boundary condition for the different
        # sides
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, bounds.west, "dir")

        # Calling helper function for assigning the Robin weight
        self.assign_robin_weight(sd=sd, bc=bc)
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        return np.zeros((self.nd, boundary_grid.num_cells)).ravel("F")

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Dirichlet BC in x, tapered in y along the west boundary."""
        values = np.zeros((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)
        t = self.time_manager.time

        xmin = self.domain.bounding_box["xmin"]
        bc_left = self.wave_function()

        # y-coordinates of west boundary cell centers
        y_west = bg.cell_centers[1, bounds.west]

        # Define the tapered box function
        y0, y1 = 0.25 * 25.0e-3, 0.75 * 25.0e-3  # support interval
        w = np.zeros_like(y_west)

        # Normalize y to support [y0, y1]
        y_norm = (y_west - y0) / (y1 - y0)
        y_norm = np.clip(y_norm, 0.0, 1.0)

        # Smooth sin^2 taper
        w = np.sin(np.pi * y_norm) ** 2

        # Apply the weighted BC
        values[0][bounds.west] += w * bc_left[0](xmin, t)
        return values.ravel("F")

    def fracture_normal_stiffness(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        stiffness = 2.0e11
        return pp.ad.Scalar(stiffness, "fracture_normal_stiffness")

    def fracture_tangential_stiffness(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        stiffness = 2.0e11
        return pp.ad.Scalar(stiffness, "fracture_tangential_stiffness")

    def maximum_elastic_fracture_opening(self, subdomains: list[pp.Grid]) -> np.ndarray:

        u_max_lower = self.solid.maximum_elastic_fracture_opening
        u_max_upper = u_max_lower * 20.0
        values = []
        for sd in subdomains:
            cc_y = sd.cell_centers[1, :]

            if np.all(cc_y > 12.5e-3):
                values.append(np.full(sd.num_cells, u_max_upper))
            elif np.all(cc_y < 12.5e-3):
                values.append(np.full(sd.num_cells, u_max_lower))
            else:
                raise ValueError("Subdomain spans both upper and lower regions")

        concat = np.concatenate(values)
        return pp.ad.DenseArray(concat, "maximum_elastic_fracture_opening")

    def data_to_export(self):
        """"""
        data = super().data_to_export()

        sd_frac_list = self.mdg.subdomains(dim=self.nd - 1)

        import os

        results_dir = self.results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Clear old files at the start of simulation (time_index == 0, first fracture)
        if self.time_manager.time_index == 0:
            for frac_idx in range(len(sd_frac_list)):
                fracture_dir = os.path.join(results_dir, f"fracture_{frac_idx}")
                os.makedirs(fracture_dir, exist_ok=True)
                for filename in [
                    "displacement_jump_n.txt",
                    "displacement_jump_t.txt",
                    "fracture_cc.txt",
                    "traction_n.txt",
                    "traction_t.txt",
                    "fracture_opening.txt",
                    "slip_tendency.txt",
                ]:
                    filepath = os.path.join(fracture_dir, filename)
                    if os.path.exists(filepath):
                        os.remove(filepath)

        # Loop over all fractures
        for frac_idx, sd_frac in enumerate(sd_frac_list):
            # Create subdirectory for this fracture
            fracture_dir = os.path.join(results_dir, f"fracture_{frac_idx}")
            os.makedirs(fracture_dir, exist_ok=True)

            nd_vec_to_normal = self.normal_component([sd_frac])
            nd_vec_to_tangential = self.tangential_component([sd_frac])

            # Displacement jump
            displacement_jump = self.displacement_jump([sd_frac])
            displacement_jump_t = self.equation_system.evaluate(
                nd_vec_to_tangential @ displacement_jump
            )
            displacement_jump_n = self.equation_system.evaluate(
                nd_vec_to_normal @ displacement_jump
            )
            displacement_jump_value = self.evaluate_and_scale(
                [sd_frac], "displacement_jump", "m"
            )
            reshaped_jump = displacement_jump_value.reshape(
                (self.nd, sd_frac.num_cells), order="F"
            )

            # Traction
            traction = self.contact_traction([sd_frac])
            traction_t = self.equation_system.evaluate(nd_vec_to_tangential @ traction)
            traction_n = self.equation_system.evaluate(nd_vec_to_normal @ traction)
            approx_force = self.equation_system.evaluate(traction)
            reshaped_traction = approx_force.reshape(
                (self.nd, sd_frac.num_cells), order="F"
            )

            # Fracture opening
            gap = self.evaluate_and_scale([sd_frac], "fracture_gap", "m")
            fracture_opening = reshaped_jump[self.nd - 1, :] - gap

            # Slip tendency
            friction_coefficient = self.evaluate_and_scale(
                sd_frac, "friction_coefficient", ""
            )
            friction_coefficient = np.ones(sd_frac.num_cells) * friction_coefficient
            traction_eval = self.evaluate_and_scale(
                [sd_frac], "contact_traction", "-"
            ).reshape((self.nd, -1), order="F")
            slip_tendency = self.compute_slip_tendency(
                traction_eval, friction_coefficient, atol=1.0e-9
            )

            # File paths for this fracture
            displacement_jump_file_n = os.path.join(
                fracture_dir, "displacement_jump_n.txt"
            )
            displacement_jump_file_t = os.path.join(
                fracture_dir, "displacement_jump_t.txt"
            )
            traction_file_n = os.path.join(fracture_dir, "traction_n.txt")
            traction_file_t = os.path.join(fracture_dir, "traction_t.txt")
            fracture_cell_centers_file = os.path.join(fracture_dir, "fracture_cc.txt")
            fracture_opening_file = os.path.join(fracture_dir, "fracture_opening.txt")
            slip_tendency_file = os.path.join(fracture_dir, "slip_tendency.txt")

            # Append data at each timestep
            def write_array(file, arr):
                with open(file, "a") as f:
                    f.write(
                        np.array2string(
                            arr, threshold=np.inf, max_line_width=np.inf, separator=", "
                        )
                        + ",\n"
                    )

            write_array(displacement_jump_file_n, displacement_jump_n)
            write_array(displacement_jump_file_t, displacement_jump_t)
            write_array(traction_file_n, traction_n)
            write_array(traction_file_t, traction_t)
            write_array(fracture_opening_file, fracture_opening)
            write_array(slip_tendency_file, slip_tendency)

            if self.time_manager.time_index == 0:
                with open(fracture_cell_centers_file, "a") as f:
                    for i in range(sd_frac.cell_centers.shape[0]):
                        line = np.array2string(
                            sd_frac.cell_centers[i, :],
                            threshold=np.inf,
                            max_line_width=np.inf,
                            separator=", ",
                        )
                        f.write(line + ",\n")
        return data


class CBB(
    GeometryBoundaryConditionAndWaveFunction,
    ContactModelBartonBandisGapFunction,
):
    """"""


run = "C-BB"
u_max = 1.0e-5
A = 5.0e-5
wave_frequency = 100.0e3

final_time = 1.3e-5
num_steps = 40 * 2**2

dt = final_time / num_steps
time_manager = pp.TimeManager(
    schedule=[0.0, final_time],
    dt_init=dt,
    constant_dt=True,
)

lmbda_lime = 4.0e9
mu_limestone = 4.0e9
rho_limestone = 2600.0

solid_vals = {
    "fracture_gap": 0.0,
    "dilation_angle": 0.0,
    "friction_coefficient": 1.0,
    "shear_modulus": mu_limestone,
    "lame_lambda": lmbda_lime,
    "density": rho_limestone,
    "maximum_elastic_fracture_opening": u_max,
}
solid = pp.SolidConstants(**solid_vals)

# Include the time_manager to the model params dictionary
params = {
    "time_manager": time_manager,
    "folder_name": "simulation_example_symmetric_fractures_2d",
    "grid_type": "simplex",
    "meshing_arguments": {
        "cell_size_fracture": 0.25e-3 if COARSE else 0.1e-3,
        "cell_size_boundary": 0.75e-3 if COARSE else 0.5e-3,
        "background_transition_multiplier": 50.0,
    },
    "material_constants": {"solid": solid},
    "wave_amplitude": A,
    "wave_frequency": wave_frequency,
    "solver_statistics_file_name": "solver_statistics.json",
    "linear_solver": {
        # "options": 
        #     {"gmres": {
        #         "ksp_monitor": None,
        #     }},
    },
}


if __name__ == "__main__":
    model = CBB(params)
    model.results_dir = "simulation_example_results_2d"
    other_params = {
        "progressbars": True,
        "nl_max_iterations": 30,
        "nl_convergence_inc_atol": 1.0e-10,
        "nl_convergence_res_atol": 1.0e-10,
    }
    runner = pp.ModelRunner(model, other_params)
    runner.run()
    
    if SAVE_FIGURES:
        import plot_simulation_example_symmetric_fractures_2d
