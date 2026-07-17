"""This file contains model setups for fracture deformation with set boundary
conditions, initial conditions, and so on. The setups are used for both convergence
analyses and for simulation examples.

The models are used for convergence analysis:
* SpringTypeBartonBandisConvergenceSetup:
    Barton-Bandis spring type deformation model (convergence of transmission
    coefficient)
* SpringTypeBartonBandis:
    Barton-Bandis spring type deformation model (used in model comparison example)
* SpringTypeLinear:
    Linear spring type deformation model (used in model comparison example)
* ContactModelBartonBandisGapFunction
    Fracture contact mechanics with Barton-Bandis elastic normal deformation (self
    convergence). Radial return contact formulation.
* ContactModelLinearGapFunction
    Fracture contact mechanics with linear elastic normal deformation (self
    convergence). Radial return contact formulation.

The ContactModelBartonBandisGapFunction model is also used in a simulation
example setting:
* Geometrically symmetric fracture network, but different Barton-Bandis parameters for
    the different fractures (runscript_example_symmetric_fractures_2d.py and
    runscript_example_symmetric_fractures_3d.py

"""

import numpy as np
import porepy as pp
import sympy as sym

from models import (
    DynamicMomentumBalanceRadialReturn,
    DynamicMomentumBalanceBartonBandisSpringModel,
    DynamicMomentumBalanceLinearSpringModel,
)

from pp_solvers import IterativeSolverMixin

class TheoreticalConstants:
    @property
    def discontinuity_location(self) -> float:
        """Location on the x-axis of the vertical split of the simulation domain.

        Returns:
            The discontinuity location.

        """
        return self.params.get("discontinuity_location", None)

    def wave_frequency(self) -> float:
        """Frequency of the incidence wave [s^-1].

        Returns:
            The wave frequency.

        """
        return self.params.get("wave_frequency", None)


class BoundaryConditions:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Method for assigning boundary condition type.

        Assigns the following boundary condition types:
            * North and south: Roller boundary (Neumann in x-direction and Dirichlet in
              y-direction)
            * West: Dirichlet
            * East: Robin

        Parameters:
            sd: The subdomain whose bc type is assigned.

        Return:
            The boundary condition object.

        """
        # Fetch boundary sides and assign type of boundary condition for the different
        # sides
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, bounds.all_bf, "dir")

        # # East side: Absorbing
        bc.is_dir[:, bounds.east] = False
        bc.is_rob[:, bounds.east] = True

        # North side: Roller
        bc.is_dir[0, bounds.north + bounds.south] = False
        bc.is_neu[0, bounds.north + bounds.south] = True

        # Calling helper function for assigning the Robin weight
        self.assign_robin_weight(sd=sd, bc=bc)
        bc.internal_to_dirichlet(sd)
        return bc

    def wave_function(self):
        """Boundary condition function.

        Returns a lambdified function which is mainly used for setting the time
        dependent Dirichlet boundary condition on the west boundary. The function is a
        sin^2 wave, which allows for a smooth start of the wave.

        Returns:
            A list with three components where the first component (x-direction)
            represents the wave function. The other two components are just zero, where
            two zero-components are needed due to 3D simulations also utilizing the
            model setups found in this file.

        """
        x, t = sym.symbols("x t")
        omega = 2 * sym.pi * self.wave_frequency()
        c_p = self.primary_wave_speed(is_scalar=True)
        A = self.params["wave_amplitude"]

        u_left = A * sym.sin(omega * (t - x / c_p)) ** 2

        u_left_func = sym.lambdify((x, t), u_left, "numpy")
        return [u_left_func, 0, 0]

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Method for assigning Neumann and Robin boundary condition values.

        Specifically, for the Robin values, this method assigns the values corresponding
        to absorbing boundary conditions with a second order approximation of u_t in:

            sigma * n + alpha * u_t = G

        Robin/Absorbing boundaries are employed for the east boundary. Zero Neumann
        values are assigned for the north and south boundary.

        Parameters:
            boundary_grid: The boundary grids on which to define boundary conditions.

        Returns:
            Array of boundary values.

        """
        if boundary_grid.dim != (self.nd - 1):
            return np.array([])

        data = self.mdg.boundary_grid_data(boundary_grid)
        sd = boundary_grid.parent
        boundary_faces = sd.get_boundary_faces()
        name = "boundary_displacement_values"

        if self.time_manager.time_index == 0:
            return self.ic_values_bc(boundary_grid)

        displacement_values_0 = pp.get_solution_values(
            name=name, data=data, time_step_index=0
        )
        displacement_values_1 = pp.get_solution_values(
            name=name, data=data, time_step_index=1
        )

        # According to the expression for the absorbing boundaries we have a coefficient
        # 2 in front of the values u_(n-1) and -0.5 in front of u_(n-2):
        displacement_values = 2 * displacement_values_0 - 0.5 * displacement_values_1

        # Transposing and reshaping displacement values to prepare for broadcasting
        displacement_values = displacement_values.reshape(
            boundary_grid.num_cells, self.nd, 1
        )

        # Assembling the vector representing the RHS of the Robin conditions
        total_coefficient_matrix = self.total_coefficient_matrix(sd=sd)
        robin_rhs = np.matmul(total_coefficient_matrix, displacement_values).squeeze(-1)
        robin_rhs *= sd.face_areas[boundary_faces][:, None]

        # Values corresponding to the right-hand side of the absorbing boundaries for
        # all boundary sides.
        robin_rhs = robin_rhs.T

        # As the same method (bc_values_stress) is used for assigning Neumann and Robin
        # (Absorbing) boundaries, we need to set zero-values where we have assigned the
        # Neumann boundary type.
        boundary_sides = self.domain_boundary_sides(sd)
        for direction in ["north", "south"]:
            inds = np.where(getattr(boundary_sides, direction))[0]
            inds = np.where(np.isin(boundary_faces, inds))[0]
            robin_rhs[:, inds] *= 0

        return robin_rhs.ravel("F")

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Method for setting Dirichlet boundary values.

        Sets a time dependent condition in the x-direction of the west boundary. Zero
        elsewhere. The boundary value function is determined by the method
        wave_function().

        Parameters:
            bg: Boundary grid whose boundary displacement value is to be set.

        Returns:
            An array with the displacement boundary values at time t.

        """
        values = np.zeros((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)
        t = self.time_manager.time

        xmin = self.domain.bounding_box["xmin"]

        bc_left = self.wave_function()
        values[0][bounds.west] += np.ones(len(values[0][bounds.west])) * bc_left[0](
            xmin, t
        )
        return values.ravel("F")

    def evaluate_mechanics_source(self, f: list, sd: pp.Grid, t: float) -> np.ndarray:
        vals = np.zeros((self.nd, sd.num_cells))
        return vals.ravel("F")


class InitialConditions:
    def initial_condition_value_function_boundary(
        self, bg: pp.BoundaryGrid, t: float
    ) -> np.ndarray:
        """Helper function to set initial values for the absorbing boundary.

        The initial values for the absorbing boundary condition are zero for all
        simulations which utilize the models found here.

        Parameters:
            bg: The boundary grid where the initial values are to be defined.
            t: The time which the values are to be defined for. Typically t = 0 or t =
                -dt, as we set initial values for the boundary condition both at initial
                time and one time-step back in time.

        Returns:
            An array of the initial boundary values.

        """
        sd = bg.parent
        bc_vals = np.zeros((sd.dim, sd.num_faces))

        # Mapping the face-wise values of the parent grid onto the boundary grid.
        bc_vals = bg.projection(self.nd) @ bc_vals.ravel("F")
        return bc_vals

    def ic_values_bc(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Method for setting initial boundary values for 0th and -1st time step.

        These values are used to initialize the absorbing boundary conditions.

        Parameters:
            bg: Boundary grid whose boundary displacement value is to be set.

        Returns:
            An array with the initial displacement boundary values.

        """
        dt = self.time_manager.dt
        vals_0 = self.initial_condition_value_function_boundary(bg=bg, t=0)
        vals_1 = self.initial_condition_value_function_boundary(bg=bg, t=0 - dt)

        data = self.mdg.boundary_grid_data(bg)

        # The values for the 0th and -1th time step are to be stored
        pp.set_solution_values(
            name="boundary_displacement_values",
            values=vals_0,
            data=data,
            time_step_index=0,
        )
        pp.set_solution_values(
            name="boundary_displacement_values",
            values=vals_1,
            data=data,
            time_step_index=1,
        )
        return vals_0

    def ic_values_displacement(self, sd: pp.Grid) -> np.ndarray:
        """Compute the initial displacement."""
        return np.zeros((self.nd, sd.num_cells)).ravel("F")

    def ic_values_velocity(self, sd: pp.Grid) -> np.ndarray:
        """Compute the initial velocity."""
        return np.zeros((self.nd, sd.num_cells)).ravel("F")

    def ic_values_acceleration(self, sd: pp.Grid) -> np.ndarray:
        """Compute the initial acceleration."""
        return np.zeros((self.nd, sd.num_cells)).ravel("F")

    def ic_values_contact_traction(self, sd: pp.Grid) -> np.ndarray:
        """Initial contact traction values on the fracture."""
        return np.zeros((self.nd, sd.num_cells)).ravel("F")


class MethodsForBartonBandisConvergenceSetup:
    """Mixin for various methods unique for the Barton-Bandis convergence setup.

    Contains:
        * Methods for setting the geometry and domain.
        * Method for setting the time dependent boundary condition function.
        * Method for computing the theoretical transmission coefficient and its error.

    """

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Method for setting Dirichlet boundary values.

        Sets a time dependent condition in the x-direction of the west boundary. Zero
        elsewhere. The boundary value function is determined by the method
        wave_function().

        Parameters:
            bg: Boundary grid whose boundary displacement value is to be set.

        Returns:
            An array with the displacement boundary values at time t.

        """
        values = np.zeros((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)
        t = self.time_manager.time

        xmin = self.domain.bounding_box["xmin"]
        # Stop after one period of the incidence wave
        if self.time_manager.time <= 5.0e-6:
            bc_left = self.wave_function()
            values[0][bounds.west] += np.ones(len(values[0][bounds.west])) * bc_left[0](
                xmin, t
            )
        return values.ravel("F")

    # Geometry and domain
    def set_fractures(self) -> None:
        """Setting fractures."""
        frac_1_points = self.units.convert_units(
            np.array(
                [
                    [self.discontinuity_location, self.discontinuity_location],
                    [0.0, 25.0e-3],
                ]
            ),
            "m",
        )
        frac_1 = pp.LineFracture(frac_1_points)
        self._fractures = [frac_1]

    def set_domain(self) -> None:
        """Domain of the problem."""
        x = self.units.convert_units(50.0e-3, "m")
        y = self.units.convert_units(25.0e-3, "m")
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x, "ymin": 0, "ymax": y}
        self._domain = pp.Domain(box)

    # Boundary condition wave function
    def wave_function(self):
        """Boundary condition function for the Barton-Bandis convergence setup."""
        x, t = sym.symbols("x t")
        A_0 = self.params["wave_amplitude"]
        omega = 2 * sym.pi * self.wave_frequency()
        u_left = A_0 * (sym.sin(omega * t)) ** 4
        u_left_func = sym.lambdify((x, t), u_left, "numpy")
        return [u_left_func, 0]

    # Data exportation and theoretical transmission coefficient computation
    def data_to_export(self):
        """Method for computing the transmission coefficient and its error."""
        data = super().data_to_export()
        sd = self.mdg.subdomains(dim=2)[0]
        sd_frac = self.mdg.subdomains(dim=1)[0]

        x = sd.cell_centers[0, :]

        L = self.discontinuity_location
        right_layer = x > L

        if self.time_manager.final_time_reached():
            velocity = self.velocity_time_dep_array([sd])
            velocity_values = self.equation_system.evaluate(velocity)
            velocity_reshaped = velocity_values.reshape(
                (self.nd, sd.num_cells), order="F"
            )
            max_velocity = np.max(velocity_reshaped[0][right_layer])
            U = self.params["wave_amplitude"]
            f = self.params["wave_frequency"]

            max_incident_velocity = 3 * np.sqrt(3) / 2 * np.pi * f * U

            T_numerical = max_velocity / max_incident_velocity
            print("T_numerical", T_numerical)

            _, T_theoretical = self.compute_theoretical_T()

            relative_error_T = abs(T_theoretical - T_numerical) / T_theoretical
            print(
                "Relative_error",
                relative_error_T,
            )
            print("T_theoretical", T_theoretical)
            with open(self.filename_path, "a") as file:
                num_cells = sd.num_cells
                file.write(
                    f"{num_cells}, {self.time_manager.time_index}, {relative_error_T}\n"
                )
        return data

    def compute_theoretical_T(self):
        """The theoretical transmission coefficient for a Barton-Bandis fracture.

        The transmission coefficient can be derived from the stress continuity condition
        and the displacement discontinuity condition (jump in displacement equals the
        Barton-Bandis term) across the fracture. This method provides a numerical
        approximation to the theoretical T.

        Reference:
            See Zhao and Cai (2001): Transmission of Elastic P-waves across Single
            Fractures with Nonlinear Deformational Behavior.

        """
        f = self.params["wave_frequency"]
        U = self.params["wave_amplitude"]
        u_max = self.solid.maximum_elastic_fracture_opening
        K_n = self.solid.fracture_normal_stiffness
        mu = self.solid.shear_modulus
        lmbda = self.solid.lame_lambda
        rho = self.solid.density

        c_p = np.sqrt((lmbda + 2 * mu) / (rho))
        z = rho * c_p
        omega = 2 * np.pi * f

        Te = 1 / (2 * f)

        m = 100000
        t_end = self.time_manager.time_final
        dt = Te / m
        t = np.arange(0, t_end, dt)

        x1 = self.discontinuity_location
        V_inc = 3 * np.sqrt(3) / 2 * np.pi * f * U

        def p(t):
            t0 = 0.0
            t1 = Te
            if t0 <= t <= t1:
                return (
                    4
                    * U
                    * omega
                    * (np.sin(omega * (t - t0))) ** 3
                    * np.cos(omega * (t - t0))
                )
            else:
                return 0.0

        v = np.zeros_like(t)
        v[0] = 0.0

        for i in range(1, len(t)):
            numerator = Te * (2 * v[i - 1] - 2 * p(t[i - 1] - x1 / c_p))
            denominator_1 = -(z) / (K_n + (z * v[i - 1]) / (u_max))
            denominator_2 = (z**2 * v[i - 1]) / (
                u_max * (K_n + (z * v[i - 1]) / (u_max)) ** 2
            )

            v[i] = 1 / m * numerator / (denominator_1 + denominator_2) + v[i - 1]

        T_non = np.max(v) / V_inc
        return np.max(v), T_non


class CommonMixins(
    TheoreticalConstants,
    BoundaryConditions,
    InitialConditions,
):
    """Mixins common for all model setups in this file (LM, BB, C-L and C-BB)."""


class SpringTypeBartonBandisConvergenceSetup(
    IterativeSolverMixin,
    MethodsForBartonBandisConvergenceSetup,
    CommonMixins,
    DynamicMomentumBalanceBartonBandisSpringModel,
):
    """Convergence setup for nonlinear fracture deformation (Barton-Bandis).

    Dynamic momentum balance with spring-type fracture deformation, where the spring
    deformation is given by the Barton-Bandis model.

    """


class SpringTypeBartonBandis(
    IterativeSolverMixin,
    CommonMixins,
    DynamicMomentumBalanceBartonBandisSpringModel,
):
    """Dynamic momentum balance with spring type fracture deformation: Barton-Bandis.

    This model implements the Barton-Bandis model for elastic normal fracture
    deformation.

    """

class SpringTypeLinear(
    IterativeSolverMixin,
    CommonMixins,
    DynamicMomentumBalanceLinearSpringModel,
):
    """Dynamic momentum balance with spring type fracture deformation: Linear.

    This model implements the Linear model for elastic normal fracture deformation.

    """


class ContactModelBartonBandisGapFunction(
    IterativeSolverMixin,
    CommonMixins,
    DynamicMomentumBalanceRadialReturn,
):
    """Dynamic momentum balance with fracture contact mechanics, Barton-Bandis version.

    The gap function for the elastic normal deformation is given by the Barton-Bandis
    model, as elaborated in the method elastic_normal_fracture_deformation().

    """

    def elastic_normal_fracture_deformation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Barton-Bandis model for elastic normal deformation of a fracture [m].

        The model computes an increase in the normal opening as a function of the
        contact traction and material constants as elaborated in the class
        documentation.

        The returned value depends on the value of the solid constant
        maximum_elastic_fracture_opening. If its value is zero, the Barton-Bandis model
        is void, and the method returns a hard-coded pp.ad.Scalar(0) to avoid zero
        division. Otherwise, an operator which implements the Barton-Bandis model is
        returned. The special treatment amounts to a continuous extension in the limit
        of zero maximum fracture opening.

        The implementation is based on the paper

        References:
            Fundamentals of Rock Joint Deformation, by S.C. Bandis, A.C.Lumdsen, N.R.
            Barton, International Journal of Rock Mechanics & Mining Sciences, 1983,
            Link: https://doi.org/10.1016/0148-9062(83)90595-8.

            See in particular Equations (8)-(9) (page 10) in that paper.

        Parameters:
            subdomains: List of fracture subdomains.

        Raises:
            ValueError: If the maximum fracture opening is negative.

        Returns:
            The elastic fracture opening, as computed by the Barton-Bandis model.

        """
        original_equation = super().elastic_normal_fracture_deformation(subdomains)

        # The maximum opening of the fracture.
        maximum_opening = self.maximum_elastic_fracture_opening(subdomains)
        elastic_opening = original_equation - maximum_opening
        elastic_opening.set_name("Barton-Bandis_elastic_opening")
        return elastic_opening


class ContactModelLinearGapFunction(
    IterativeSolverMixin,
    CommonMixins,
    DynamicMomentumBalanceRadialReturn,
):
    """Dynamic momentum balance with fracture contact mechanics, linear gap version.

    The gap function for the elastic normal deformation is given by a linear function,
    as elaborated in the method elastic_normal_fracture_deformation().

    """

    def elastic_normal_fracture_deformation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """The elastic normal fracture deformation [m].

        The elastic normal fracture deformation is the normal component of the
        displacement jump, which is the solution to the contact mechanics problem in the
        absence of plastic deformation.

        The elastic normal deformation is given by
        .. math::
            u_n = \frac{t_n}{K_n} = Z_n * t_n,
        where :math:`t_n` is the normal component of the contact traction and
        :math:`K_n` is the normal stiffness. Z_n is the normal fracture compliance,
        which is the inverse of the normal fracture stiffness. If the stiffness is
        negative, the deformation is set to zero, thus avoiding using :math:`K_n->inf`
        to represent no normal deformation.

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Operator representing the elastic normal fracture deformation.

        """
        nd_vec_to_normal = self.normal_component(subdomains)
        t_n = nd_vec_to_normal @ self.contact_traction(subdomains)
        stiffness = self.fracture_normal_stiffness(subdomains)

        # Retrieve the *unscaled* stiffness value for the check below.
        stiffness_value = self.units.convert_units(
            self.equation_system.evaluate(stiffness),
            "Pa*m^-1",
            to_si=True,
        )
        if np.any(np.isclose(stiffness_value, -1.0, atol=1e-12, rtol=1e-12)):
            # Stiffness=-1 indicates no elastic normal deformation. Small tolerances
            # are used to avoid numerical issues, but allowing for a float value.
            num_cells = sum(sd.num_cells for sd in subdomains)
            zero_u_n = pp.ad.DenseArray(np.zeros((self.nd - 1) * num_cells))
            zero_u_n.set_name("zero_elastic_normal_fracture_deformation")
            return zero_u_n

        # Since contact traction is nondimensional, the stiffness must be scaled by the
        # characteristic contact traction.
        scaled_stiffness = stiffness / self.characteristic_contact_traction(subdomains)
        u_n = t_n / scaled_stiffness
        u_n.set_name("elastic_normal_fracture_deformation")
        return u_n
