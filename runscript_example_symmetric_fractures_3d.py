import numpy as np
import porepy as pp
import sympy as sym
from numpy.typing import NDArray
from models_nonlinear_fracture_deformation import ContactModelBartonBandisGapFunction
import logging
 
logger = logging.getLogger(__name__)
 
logging.basicConfig(level=logging.INFO)

class GeometryBoundaryConditionAndWaveFunction:
    def fracture_network_2d(self) -> None:
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

        domain_2d = pp.Domain(self.box_2d())
        return pp.create_fracture_network(
            [frac_1, frac_2, frac_3, frac_4, frac_5, frac_6], domain_2d
        )

    def box_2d(self) -> dict:
        x = self.units.convert_units(25.0e-3, "m")
        y = self.units.convert_units(25.0e-3, "m")
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x, "ymin": 0, "ymax": y}
        return box

    def set_domain(self) -> None:
        """Domain of the problem."""
        z = self.units.convert_units(25.0e-3, "m")
        box = self.box_2d()
        box.update({"zmin": 0, "zmax": z})
        self._domain = pp.Domain(box)

    def create_mdg(self) -> None:
        """Set the mixed-dimensional grid from the domain, fracture network and meshing
        arguments.
        """
        self.mdg = pp.create_mdg(
            self.grid_type(),
            self.meshing_arguments(),
            self.fracture_network_2d(),
            **self.meshing_kwargs(),
        )
        height = self.domain.bounding_box["xmax"]
        cell_size = self.meshing_arguments()["cell_size_fracture"]
        n_layers = int(np.round(height / cell_size))
        z = np.linspace(0, height, n_layers + 1)

        self.mdg, _ = pp.grid_extrusion.extrude_mdg(self.mdg, z)
        self.mdg.compute_geometry()
        self.mdg.set_boundary_grid_projections()

        self.nd: int = self.mdg.dim_max()

        # Create projections between local and global coordinates for fracture grids.
        pp.set_local_coordinate_projections(self.mdg)

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
        z_west = bg.cell_centers[2, bounds.west]

        # Define the tapered box function
        y0, y1 = 0.25 * 25.0e-3, 0.75 * 25.0e-3
        z0, z1 = 0.25 * 25.0e-3, 0.75 * 25.0e-3
        w = np.zeros_like(y_west)

        # Normalize y to support [y0, y1]
        y_norm = (y_west - y0) / (y1 - y0)
        y_norm = np.clip(y_norm, 0.0, 1.0)

        # Same for z to support [z0, z1]
        z_norm = (z_west - z0) / (z1 - z0)
        z_norm = np.clip(z_norm, 0.0, 1.0)

        # Smooth sin^2 taper
        wy = np.sin(np.pi * y_norm) ** 2
        wz = np.sin(np.pi * z_norm) ** 2

        w = wy * wz

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

    def data_to_export(self) -> list:
        """Returns data for exporting.

        Returns:
            A list of tuples (subdomain, variable name, variable values).
        """
        # Start with data from super class. This includes standard variables.
        data = super().data_to_export()  # type: ignore[misc]
        # Add the three fracture-specific vector quantities.
        sds = self.mdg.subdomains(dim=self.nd - 1)
        cell_offsets_nd = np.cumsum([0] + [sd.num_cells * self.nd for sd in sds])
        cell_offsets = np.cumsum([0] + [sd.num_cells for sd in sds])
        displacement_jump = self.evaluate_and_scale(sds, "displacement_jump", "m")
        char = self.evaluate_and_scale(sds, "characteristic_contact_traction", "Pa")
        friction_coefficient = self.evaluate_and_scale(sds, "friction_coefficient", "")

        # Both characteristic traction and friction coefficients are frequently floats.
        # Ensure they are arrays to allow element-wise operations below, thus covering
        # the case of spatially homogeneous quantities. Heterogeneous quantities are
        # at least possible for the friction coefficient.
        size = sum([sd.num_cells for sd in sds])

        def ensure_array(
            quantity: NDArray | float,
        ) -> NDArray:
            if isinstance(quantity, float):
                # Cast to cell-wise array.
                return quantity * np.ones(size)
            else:
                return quantity

        char = ensure_array(char)
        friction_coefficient = ensure_array(friction_coefficient)
        traction = self.evaluate_and_scale(sds, "contact_traction", "-").reshape(
            (self.nd, -1), order="F"
        )
        # Compute apertures, which are scalar quantities.
        cell_offsets = np.cumsum([0] + [sd.num_cells for sd in sds])
        apertures = self.evaluate_and_scale(sds, "aperture", "m")
        slip_tendency = self.compute_slip_tendency(
            traction, friction_coefficient, atol=1.0e-7
        )

        # Loop over the fracture subdomains.
        for id, sd in enumerate(sds):
            # Export the displacement jump.
            data.append(
                (
                    sd,
                    "displacement_jump",
                    displacement_jump[cell_offsets_nd[id] : cell_offsets_nd[id + 1]],
                )
            )
            # Export the slip tendency, defined as the ratio of the shear traction to
            # the normal traction.
            data.append(
                (
                    sd,
                    "slip_tendency",
                    slip_tendency[cell_offsets[id] : cell_offsets[id + 1]],
                )
            )
            # Rescale traction by characteristic contact traction.
            traction_loc = traction[:, cell_offsets[id] : cell_offsets[id + 1]]
            traction_loc *= char[cell_offsets[id] : cell_offsets[id + 1]]
            data.append((sd, "contact_traction_in_Pa", traction_loc.ravel("F")))

            data.append(
                (
                    sd,
                    "aperture",
                    apertures[cell_offsets[id] : cell_offsets[id + 1]],
                )
            )
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
    "folder_name": "simulation_example_symmetric_fractures_3d",
    "grid_type": "simplex",
    "meshing_arguments": {
        "cell_size_fracture": 0.5e-3,
        "cell_size_boundary": 1.0e-3,
        "background_transition_multiplier": 20.0,
    },
    "material_constants": {"solid": solid},
    "wave_amplitude": A,
    "wave_frequency": wave_frequency,
    "solver_statistics_file_name": "solver_statistics.json",
}

model = CBB(params)
model.file_suffix = "test"
other_params = {
    "progressbars": True,
    "nl_max_iterations": 30,
    "nl_convergence_inc_atol": 1.0e-10,
    "nl_convergence_res_atol": 1.0e-10,
}
runner = pp.ModelRunner(model, other_params)
runner.run()
