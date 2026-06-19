import numpy as np
import porepy as pp

from models_nonlinear_fracture_deformation import (
    ContactModelBartonBandisGapFunction,
    ContactModelLinearGapFunction,
)


class GeometryBoundaryConditionAndWaveFunction:
    def set_fractures(self) -> None:
        """Setting fractures."""

        # Original points in meters (2x2 array: [[x0, x1], [y0, y1]])
        frac_1_points = self.units.convert_units(
            np.array([[12.5e-3, 12.5e-3], [4.5e-3, 20.5e-3]]), "m"
        )

        # Rotate by -pi/6 while keeping midpoint the same
        rotated_points = self.rotate_line_keep_midpoint(frac_1_points, theta=-np.pi / 6)

        # Create the line fracture
        frac_1 = pp.LineFracture(rotated_points)
        self._fractures = [frac_1]

    def rotate_line_keep_midpoint(self, points, theta):
        """Rotate a line by angle theta (radians).

        First the line is rotated, and then it is translated to restore the original
        midpoint.

        Parameters:
            points: An array with the line endpoint coordinates ([[x0, x1], [y0, y1]]).
            theta: rotation angle in radians.

        Returns: 2x2 array of rotated + translated line
        """
        # Original midpoint
        M_orig = np.mean(points, axis=1, keepdims=True)

        # Rotation matrix
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        # Rotate points
        rotated = R @ points  # matrix multiplication

        # Rotated midpoint
        M_rot = np.mean(rotated, axis=1, keepdims=True)

        # Translation vector to restore original midpoint
        translation = M_orig - M_rot

        # Apply translation
        translated = rotated + translation
        return translated

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

    def data_to_export(self):
        """"""
        data = super().data_to_export()
        # Export to .txt files for the reference model
        if self.params["reference_flag"]:
            sd_frac = self.mdg.subdomains(dim=1)[0]
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

            traction_n_nondim = self.equation_system.evaluate(
                nd_vec_to_normal @ traction
            )
            traction_t_nondim = self.equation_system.evaluate(
                nd_vec_to_tangential @ traction
            )

            traction_scaled = traction * self.characteristic_contact_traction([sd_frac])
            traction_t_Pa = self.equation_system.evaluate(
                nd_vec_to_tangential @ traction_scaled
            )
            traction_n_Pa = self.equation_system.evaluate(
                nd_vec_to_normal @ traction_scaled
            )

            # Fracture opening
            gap = self.evaluate_and_scale([sd_frac], "fracture_gap", "m")
            fracture_opening = reshaped_jump[self.nd - 1, :] - gap

            # Slip tendency
            friction_coefficient = self.evaluate_and_scale(
                sd_frac, "friction_coefficient", ""
            )
            friction_coefficient = np.ones(sd_frac.num_cells) * friction_coefficient
            traction = self.evaluate_and_scale(
                [sd_frac], "contact_traction", "-"
            ).reshape((self.nd, -1), order="F")
            slip_tendency = self.compute_slip_tendency(
                traction, friction_coefficient, atol=5.0e-7
            )

            import os

            results_dir = "convergence_analysis_results"
            model_tag = self.params["model_tag"]
            model_dir = os.path.join(results_dir, model_tag)
            os.makedirs(model_dir, exist_ok=True)

            # displacement_jump_file = os.path.join(model_dir, "displacement_jump.txt")
            displacement_jump_file_n = os.path.join(
                model_dir, "displacement_jump_n.txt"
            )
            displacement_jump_file_t = os.path.join(
                model_dir, "displacement_jump_t.txt"
            )
            traction_file_n_nondim = os.path.join(model_dir, "traction_n_nondim.txt")
            traction_file_t_nondim = os.path.join(model_dir, "traction_t_nondim.txt")

            traction_file_n_Pa = os.path.join(model_dir, "traction_n.txt")
            traction_file_t_Pa = os.path.join(model_dir, "traction_t.txt")

            fracture_cell_centers_file = os.path.join(model_dir, "fracture_cc.txt")
            fracture_opening_file = os.path.join(model_dir, "fracture_opening.txt")
            slip_tendency_file = os.path.join(model_dir, "slip_tendency.txt")

            # Make sure NumPy never truncates arrays globally
            np.set_printoptions(threshold=np.inf, linewidth=np.inf)

            def write_array(file, arr):
                with open(file, "a") as f:
                    f.write(np.array2string(arr) + ",\n")

            # Append data at each timestep
            write_array(displacement_jump_file_n, displacement_jump_n)
            write_array(displacement_jump_file_t, displacement_jump_t)

            write_array(traction_file_n_nondim, traction_n_nondim)
            write_array(traction_file_t_nondim, traction_t_nondim)

            write_array(traction_file_n_Pa, traction_n_Pa)
            write_array(traction_file_t_Pa, traction_t_Pa)

            write_array(fracture_opening_file, fracture_opening)
            write_array(slip_tendency_file, slip_tendency)

            # Save cell centers only once
            if self.time_manager.time_index == 0:
                write_array(fracture_cell_centers_file, sd_frac.cell_centers)
        return data


class SelfConvergenceCBB(
    GeometryBoundaryConditionAndWaveFunction, ContactModelBartonBandisGapFunction
):
    """Model setup for the self convergence analysis: Barton-Bandis gap function."""


class SelfConvergenceCL(
    GeometryBoundaryConditionAndWaveFunction, ContactModelLinearGapFunction
):
    """Model setup for the self convergence analysis: Linear gap function."""
