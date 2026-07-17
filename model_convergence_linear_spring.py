"""This file contains the model setup for the linear spring-type fracture deformation
convergence analyses. All initial conditions, boundary conditions, geometry and so on
is set. The model is set up to be used in runscript_convergence_linear_spring.py.

"""

import numpy as np
import porepy as pp
import sympy as sym
from porepy.applications.convergence_analysis import ConvergenceAnalysis

from models import DynamicMomentumBalanceLinearSpringModel
from pp_solvers import IterativeSolverMixin

class Geometry:
    def set_fractures(self) -> None:
        """Setting fractures.

        There is one vertical fracture in the domain which is located at a pre-described
        "discontinuity location". The fracture spans from the bottom to the top of the
        domain.

        """
        # First fracture (original)
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
        """Domain for the problem."""
        x = self.units.convert_units(50.0e-3, "m")
        y = self.units.convert_units(25.0e-3, "m")
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x, "ymin": 0, "ymax": y}
        self._domain = pp.Domain(box)

    def set_polygons(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Define region for different stiffness tensor by lines.

        Returns the lines which surrounds the region to the right of the fracture. The
        lines are named west, north, east and south based on the side of the region they
        are on. The west line corresponds (geometrically) exactly to the fracture.

        Returns:
            A tuple containing the lines west, north, east and south which makes up the
            polygon around the region where a different stiffness tensor is to be
            defined.

        """
        if type(self.discontinuity_location) is list:
            L = self.discontinuity_location[0]
            W = self.discontinuity_location[1]
        else:
            L = self.discontinuity_location
            W = self.domain.bounding_box["xmax"]

        H = self.domain.bounding_box["ymax"]
        west = np.array([[L, L], [0.0, H]])
        north = np.array([[L, W], [H, H]])
        east = np.array([[W, W], [H, 0.0]])
        south = np.array([[W, L], [0.0, 0.0]])
        return west, north, east, south


class MiscellaneousConstants:
    @property
    def compressive_setup(self) -> bool:
        """Whether we are in the compressive setup or the shear setup.

        * Compressive setup: The wave is compressive, meaning that all displacements
            are in the same direction as the wave propagates (which is hard-coded to be positive x-direction).
        * Shear setup: The wave is shear, meaning that all displacements are in the
            direction normal to the wave propagation direction. As the wave is hard-coded to propagate in the positive x-direction, the normal direction is y.

        Returns:
            True or False depending on whether we are in the compressive setup or the shear setup.

        """
        return self.params.get("compressive_setup", True)

    @property
    def heterogeneity_factor(self) -> float:
        """Factor determining how strong heterogeneity we have in a domain.

        Determines the factor that the material parameters to the left of
        self.heterogeneity_location should differ from the material parameters to the
        right of self.heterogeneity_location. The default value is 1.0, which means that the materials are identical on both sides of the fracture.

        Returns:
            The heterogeneity factor.

        """
        return self.params.get("heterogeneity_factor", 1.0)

    @property
    def discontinuity_location(self) -> float:
        """Location on the x-axis of the vertical split of the simulation domain.

        Returns:
            The discontinuity location.

        """
        return self.params.get("discontinuity_location", 25.0e-3)

    def wave_frequency(self) -> float:
        """Frequency of the incidence wave [s^-1].

        Returns:
            The wave frequency.

        """
        return self.params.get("wave_frequency", 0.0)

    def vector_valued_mu_lambda_rho(self) -> None:
        """Setting a vertically layered medium based on self.discontinuity_location.

        Material to the left of the discontinuity location has stiffness parameters
        corresponding to the ones set in the solid parameters (shear_modulus and
        lame_lambda). To the right of the discontinuity location, the material
        parameters are set from "solid_values_inner_region".

        """
        subdomain = self.mdg.subdomains(dim=self.nd)[0]
        x = subdomain.cell_centers[0, :]

        lmbda1 = self.solid.lame_lambda
        mu1 = self.solid.shear_modulus
        rho1 = self.solid.density

        solid_values_inner_region = self.params.get(
            "solid_values_inner_region",
            {
                "mu_parallel": self.solid.shear_modulus,
                "mu_orthogonal": self.solid.shear_modulus,
                "lambda_parallel": 0.0,
                "lambda_orthogonal": 0.0,
                "volumetric_compr_lambda": self.solid.lame_lambda,
                "rho": self.solid.density,
            },
        )

        lmbda2 = solid_values_inner_region["volumetric_compr_lambda"]
        mu2 = solid_values_inner_region["mu_parallel"]
        rho2 = solid_values_inner_region["rho"]

        lmbda_vec = np.ones(subdomain.num_cells)
        mu_vec = np.ones(subdomain.num_cells)
        rho_vec = np.ones(subdomain.num_cells)

        left_layer = x <= self.discontinuity_location
        right_layer = x > self.discontinuity_location

        lmbda_vec[left_layer] *= lmbda1
        mu_vec[left_layer] *= mu1
        rho_vec[left_layer] *= rho1

        lmbda_vec[right_layer] *= lmbda2
        mu_vec[right_layer] *= mu2
        rho_vec[right_layer] *= rho2

        self.mu_vector = mu_vec
        self.lambda_vector = lmbda_vec
        self.rho_vector = rho_vec

        self.mu_left = mu1
        self.lmbda_left = lmbda1
        self.rho_left = rho1

        self.mu_right = mu2
        self.lmbda_right = lmbda2
        self.rho_right = rho2


class ReflectionTransmissionMethods:
    def plus_pi(self) -> bool:
        """Used to determine if we need to add pi to the phase shift calculation.

        The analytical solution to the heterogeneous fracture problem is derived from
        theoretical results for the transmission (T) and reflection (R) coefficients. T
        and R are complex valued, and their argument/phase (which is relevant for the
        analytical solution), depends on the quadrant of the complex plane.

        Set a = k / (omega * Z_A) and b = k / (omega * Z_B) where Z_A and Z_B are the
        medium impedance of the left (A) and right (B) side of the fracture.

            T = (2b)/(a + b + i), R = (i + b - a)/(i + b + a).

        Writing each of T and R on the form x + yi will show that T is always in the
        fourth quadrant. On the other hand, as a, b > 0, R is in the second or the first
        first quadrant depending on the relative magnitudes of a and b.

        This method indicates whether one needs to apply a quadrant correction.

        Returns:
            A boolean value which says whether we need to apply a quadrant correction to
            R or not.

        """
        a, _ = self.stiffness_to_impedance_ratio_R()
        b, _ = self.stiffness_to_impedance_ratio_T()
        if (1 + b**2 - a**2) >= 0:
            return False
        else:
            return True

    def fracture_phase_shift_T(self) -> float:
        """Phase shift in analytical solution, T contribution

        The transmission coefficient is complex, and we use its polar form to get
        magnitude and argument/phase. The phase ensures that reflected and transmitted
        waves combine correctly in the analytical solution u(x,t).

        Returns:
            The phase shift contribution from T.

        """
        a, _ = self.stiffness_to_impedance_ratio_R()
        b, _ = self.stiffness_to_impedance_ratio_T()

        T_phase = np.arctan(-(2 * b) / (2 * b**2 + 2 * b * a))

        return T_phase

    def fracture_phase_shift_R(self) -> float:
        """Phase shift in analytical solution, R contribution

        The reflection coefficient is complex, and we use its polar form to get
        magnitude and argument/phase. The phase ensures that reflected and transmitted
        waves combine correctly in the analytical solution u(x,t).

        Returns:
            The phase shift contribution from R.

        """
        a, _ = self.stiffness_to_impedance_ratio_R()
        b, _ = self.stiffness_to_impedance_ratio_T()
        if not self.plus_pi():
            R_phase = np.arctan((2 * a) / (1 + b**2 - a**2))
        else:
            R_phase = np.pi - np.arctan((2 * a) / (1 + b**2 - a**2))
        return R_phase

    def theoretical_reflection_coefficient(self) -> tuple[float, float]:
        """Magnitude of the reflection coefficient.

        The reflection coefficient is complex, and this method computes and returns its
        magnitude. The reflection coefficient magnitude is used in the analytical
        expression of u(x, t).

        This method also fetches the wave speed in the part of the domain where we see
        reflections from the fracture, that is, on the left side of the fracture. This
        is only relevant in the case of a heterogeneous medium, as the wave speed is
        identical on the two sides of the medium if it is homogeneous.

        Returns:
            A tuple containing the magnitude of the reflection coefficient and the
            wave speed in the medium before the fracture.

        """
        a, c = self.stiffness_to_impedance_ratio_R()
        b, _ = self.stiffness_to_impedance_ratio_T()

        first_term = (1 + b**2 - a**2) / (1 + (b + a) ** 2)
        second_term = (2 * a) / (1 + (b + a) ** 2)

        theoretical_R_modulus = np.sqrt(first_term**2 + second_term**2)
        return theoretical_R_modulus, c

    def theoretical_transmission_coefficient(self) -> tuple[float, float]:
        """Magnitude of the transmission coefficient.

        The transmission coefficient is complex, and this method computes and returns
        its magnitude. The transmission coefficient magnitude is used in the analytical
        expression of u(x, t).

        This method also fetches the wave speed in the part of the domain where we see
        transmissions through the fracture, that is, on the right side of the fracture.
        This is only relevant in the case of a heterogeneous medium, as the wave speed
        is identical on the two sides of the medium if it is homogeneous.

        Returns:
            A tuple containing the magnitude of the transmission coefficient and the
            wave speed in the medium after the fracture.

        """
        a, _ = self.stiffness_to_impedance_ratio_R()
        b, c = self.stiffness_to_impedance_ratio_T()

        first_term = (2 * b**2 + 2 * b * a) / (1 + (b + a) ** 2)
        second_term = (2 * b) / (1 + (b + a) ** 2)

        theoretical_T_modulus = np.sqrt(first_term**2 + second_term**2)
        return theoretical_T_modulus, c

    def velocity_before_and_after_fracture(self) -> tuple[float, float]:
        """Method for finding the wave speed before and after the fracture.

        Returns:
            A tuple containing the velocity before and after the fracture (in that
            order).

        """
        # The wave speed methods return arrays containing the cell-wise P- or S-wave
        # speeds. Fetch the entire array to later determine which velocity is before and
        # which velocity is after the fracture.
        if self.compressive_setup:
            c_left = np.sqrt((self.lmbda_left + 2 * self.mu_left) / self.rho_left)
            c_right = np.sqrt((self.lmbda_right + 2 * self.mu_right) / self.rho_right)
            return c_left, c_right
        else:
            c_left = np.sqrt(self.mu_left / self.rho_left)
            c_right = np.sqrt(self.mu_right / self.rho_right)
            return c_left, c_right

    def stiffness_to_impedance_ratio_R(self) -> tuple[float, float]:
        """Stifness to impedance ratio and velocity before the fracture.

        For brevity, instead of computing the ratio r = k / (Z * omega) every time it is
        needed, we have this helper function for r in the domain on the left side of the
        fracture.

        Returns:
            A tuple containing the ratio between stiffness and impedace * angular
            frequency for the reflection coefficient, and the velocity before the
            fracture.

        """
        if self.compressive_setup:
            k = self.solid.fracture_normal_stiffness
        else:
            k = self.solid.fracture_tangential_stiffness
        velocity_before, _ = self.velocity_before_and_after_fracture()
        Z = velocity_before * self.rho_left
        f = self.wave_frequency()
        omega = 2 * np.pi * f

        a = k / (Z * omega)
        return a, velocity_before

    def stiffness_to_impedance_ratio_T(self) -> tuple[float, float]:
        """Stifness to impedance ratio and velocity after the fracture.

        For brevity, instead of computing the ratio r = k / (Z * omega) every time it is
        needed, we have this helper function for r in the domain on the right side of
        the fracture.

        Returns:
            A tuple containing the ratio between stiffness and impedace * angular
            frequency for the transmission coefficient, and the velocity after the
            fracture.

        """
        if self.compressive_setup:
            k = self.solid.fracture_normal_stiffness
        else:
            k = self.solid.fracture_tangential_stiffness

        _, velocity_after = self.velocity_before_and_after_fracture()
        Z = velocity_after * self.rho_right

        f = self.wave_frequency()
        omega = 2 * np.pi * f

        a = k / (Z * omega)
        return a, velocity_after


class BoundaryAndInitialConditions:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Method for assigning boundary condition type.

        Assigns the following boundary condition types:
            * North and south: Roller boundary
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

        if sd.dim == 2:
            # East side: Absorbing
            bc.is_dir[:, bounds.east] = False
            bc.is_rob[:, bounds.east] = True

            # North side: Roller. But which component is assigned stress free and which
            # component is assigned fixed displacement depends on whether we are in the
            # compressive or shear setup.
            if self.compressive_setup:
                bc.is_dir[0, bounds.north + bounds.south] = False
                bc.is_neu[0, bounds.north + bounds.south] = True
            else:
                bc.is_dir[1, bounds.north + bounds.south] = False
                bc.is_neu[1, bounds.north + bounds.south] = True

        # Calling helper function for assigning the Robin weight
        self.assign_robin_weight(sd=sd, bc=bc)
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Method for setting Dirichlet boundary values.

        Sets a time dependent condition in the x- or y-direction of the western
        boundary. Zero elsewhere.

        The boundary value function is determined by the method analytical_solution(),
        which is the known analytical solution for a 1D wave travelling in an
        inhomogeneous domain.

        Parameters:
            bg: Boundary grid whose boundary displacement value is to be set.

        Returns:
            An array with the displacement boundary values at time t.

        """
        values = np.zeros((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)
        t = self.time_manager.time

        xmin = self.domain.bounding_box["xmin"]

        bc_left, _ = self.analytical_solution()

        # Distinguish between the boundary condition values for shear and compressive
        # setup:
        component = 0 if self.compressive_setup else 1

        values[component][bounds.west] += np.ones(
            len(values[component][bounds.west])
        ) * bc_left[component](xmin, t)
        return values.ravel("F")

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

    def initial_condition_value_function_boundary(
        self, bg: pp.BoundaryGrid, t: float
    ) -> np.ndarray:
        """Helper function to set initial values for the absorbing boundary.

        Parameters:
            bg: The boundary grid where the initial values are to be defined.
            t: The time which the values are to be defined for. Typically t = 0 or t =
                -dt, as we set initial values for the boundary condition both at initial
                time and one time-step back in time.

        Returns:
            An array of the initial boundary values.

        """
        sd = bg.parent
        x = sd.face_centers[0, :]

        bc_vals = np.zeros((sd.dim, sd.num_faces))

        boundary_sides = self.domain_boundary_sides(sd)
        inds_east = np.where(boundary_sides.east)[0]
        _, displacement_function_right = self.analytical_solution()

        # East
        component = 0 if self.compressive_setup else 1
        bc_vals[component, :][inds_east] = displacement_function_right[component](
            x[inds_east], t
        )

        # Mapping the face-wise values of the parent grid onto the boundary grid.
        bc_vals = bg.projection(self.nd) @ bc_vals.ravel("F")
        return bc_vals

    def _compute_initial_condition(
        self, return_dt: bool = False, return_ddt: bool = False
    ) -> np.ndarray:
        """Helper function to compute displacement, velocity or acceleration.

        This function fetches the analytical displacement/velocity/acceleration
        (depending on the values of return_dt and return_ddt) and evaluates it at the
        cell centers of the subdomain for the initial time. Note that default behavior
        is to compute initial displacements (return_dt and return_ddt are False).

        Parameters:
            return_dt: If True, the function computes initial velocity. return_ddt: If
            True, the function computes initial acceleration.

        Raises:
            ValueError: If both return_dt and return_ddt are True.

        Returns:
            An array of the initial condition values for either displacement, velocity
            or acceleration depending on parameter input.

        """
        if return_dt and return_ddt:
            raise ValueError(
                "Both return_dt and return_ddt cannot be True at the same time."
                "\n Please set neither or only one of them equal to True."
            )

        sd = self.mdg.subdomains()[0]
        x = sd.cell_centers[0, :]
        t = self.time_manager.time

        L = self.discontinuity_location
        left_layer = x <= L
        right_layer = x > L

        vals = np.zeros((self.nd, sd.num_cells))

        left_solution, right_solution = self.analytical_solution(
            return_dt=return_dt, return_ddt=return_ddt
        )

        component = 0 if self.compressive_setup else 1
        vals[component, left_layer] = left_solution[component](x[left_layer], t)
        vals[component, right_layer] = right_solution[component](x[right_layer], t)
        return vals.ravel("F")

    def ic_values_interface_displacement(self, intf: pp.MortarGrid) -> np.ndarray:
        """Initial values for interface displacement.

        Parameters:
            intf: The interface grid where the initial displacement values are to be
                computed.

        Returns:
            An array with the interface displacement initial values.

        """
        direction_vec = np.array([1.0, 0.0, 0.0])
        indices_positive_side, indices_negative_side, _ = pp.sides_of_fracture(
            intf, self.mdg.subdomains(dim=self.nd)[0], direction_vec
        )
        vals = np.zeros((self.nd, intf.num_cells))

        # The direction vector points in the same direction as the normal vector of the
        # left side of the fracture. Thus:
        indices_left_side = indices_positive_side
        indices_right_side = indices_negative_side

        L = self.discontinuity_location
        sol_left, sol_right = self.analytical_solution()

        flag_right = np.isin(np.arange(intf.num_cells), indices_right_side)
        flag_left = np.isin(np.arange(intf.num_cells), indices_left_side)

        component = 0 if self.compressive_setup else 1
        vals[component, flag_right] = sol_right[component](L, 0.0)
        vals[component, flag_left] = sol_left[component](L, 0.0)
        return vals.ravel("F")

    def ic_values_displacement(self, sd: pp.Grid) -> np.ndarray:
        """Compute the initial displacement."""
        return self._compute_initial_condition()

    def ic_values_velocity(self, sd: pp.Grid) -> np.ndarray:
        """Compute the initial velocity."""
        return self._compute_initial_condition(return_dt=True)

    def ic_values_acceleration(self, sd: pp.Grid) -> np.ndarray:
        """Compute the initial acceleration."""
        return self._compute_initial_condition(return_ddt=True)

    def ic_values_contact_traction(self, sd: pp.Grid) -> np.ndarray:
        """Initial contact traction values on the fracture.

        Initial traction is computed based on the initial interface displacement.

        """
        displacement_jump = self.displacement_jump([sd])
        t_char = self.characteristic_contact_traction([sd])
        initial_traction = (
            displacement_jump / t_char * self.solid.fracture_tangential_stiffness
        )
        initial_traction_values = self.equation_system.evaluate(initial_traction)
        return initial_traction_values


class ExactExpressionsAndEvaluationMethods:
    def analytical_solution(
        self, return_dt: bool = False, return_ddt: bool = False, lambdify: bool = True
    ) -> tuple:
        """Analytical solution for wave propagation in fractured, heterogeneous medium.

        The analytical expression for the displacement field u(x, t) is derived from
        theoretical results for waves propagating at normal incidence towards a linearly
        deforming fracture. This method defines the expression for u(x, t) and is set up
        to have one of the following as its "main" return value:
            * The symbolic expression for u(x, t) if both return_dt and return_ddt are
              False.
            * The symbolic expression for the time derivative of u(x, t) (velocity) if
              return_dt is True.
            * The symbolic expression for the second time derivative of u(x, t)
              (acceleration) if return_ddt is True.
            * Lambdified functions for u(x, t) or its derivatives if and lambdify is
              True.

        The default behavior is to return lambdified functions for u(x, t).

        Parameters:
            return_dt: If True, the function computes the time derivative of u(x, t).
            return_ddt: If True, the function computes the second time derivative of
                u(x, t).
            lambdify: If True, the function returns lambdified functions for u(x, t)
                or its derivatives. If False, it returns the sympy-expressions directly.
                This is typically used to get an expression for the analytical force on
                cell faces.

        Returns:
            Either a sympy-expression of u(x, t) and the sympy symbols x and t, or the
            lambdified expression of u(x, t) or its time derivatives. In both cases the
            return is a tuple.

        """
        if return_dt and return_ddt:
            raise ValueError(
                "Both return_dt and return_ddt cannot be True at the same time."
                "\n Please set neither or only one of them equal to True."
            )

        x, t = sym.symbols("x t")

        L = self.discontinuity_location
        T, right_speed = self.theoretical_transmission_coefficient()
        R, left_speed = self.theoretical_reflection_coefficient()
        A = self.params["wave_amplitude"]
        angular_frequency = 2 * np.pi * self.wave_frequency()

        u_left = A * sym.sin(
            angular_frequency * (t - (x - L) / left_speed)
        ) + A * R * sym.sin(
            angular_frequency * (t + (x - L) / left_speed)
            + self.fracture_phase_shift_R()
        )
        u_right = (
            A
            * T
            * sym.sin(
                angular_frequency * (t - (x - L) / right_speed)
                + self.fracture_phase_shift_T()
            )
        )
        # Compute derivatives based on function arguments
        if return_dt:
            u_left, u_right = sym.diff(u_left, t), sym.diff(u_right, t)
        elif return_ddt:
            u_left, u_right = sym.diff(u_left, t, 2), sym.diff(u_right, t, 2)

        if lambdify and self.compressive_setup:
            return [sym.lambdify((x, t), u_left, "numpy"), 0], [
                sym.lambdify((x, t), u_right, "numpy"),
                0,
            ]
        elif lambdify and not self.compressive_setup:
            return [0, sym.lambdify((x, t), u_left, "numpy")], [
                0,
                sym.lambdify((x, t), u_right, "numpy"),
            ]
        else:
            return x, t, u_left, u_right

    def exact_discontinuous_sigma(
        self, u: sym.Expr, lam: float, mu: float, x: sym.Symbol
    ) -> list:
        """Representation of the exact stress tensor given a displacement field.

        Parameters:
            u: The sympy representation of the exact displacement.
            lam: The first Lamé parameter.
            mu: The second Lamé parameter, also called shear modulus.
            x: The sympy representation of the x-coordinate.

        Returns:
            A list which represents the sympy expression of the exact stress tensor.

        """
        y = sym.symbols("y")

        if self.compressive_setup:
            u = [u, 0]
        else:
            u = [0, u]

        # Exact gradient of u and transpose of gradient of u
        grad_u = [
            [sym.diff(u[0], x), sym.diff(u[0], y)],
            [sym.diff(u[1], x), sym.diff(u[1], y)],
        ]

        grad_u_T = [[grad_u[0][0], grad_u[1][0]], [grad_u[0][1], grad_u[1][1]]]

        # Trace of gradient of u, in the linear algebra sense
        trace_grad_u = grad_u[0][0] + grad_u[1][1]

        # Exact strain (\epsilon(u))
        strain = 0.5 * np.array(
            [
                [grad_u[0][0] + grad_u_T[0][0], grad_u[0][1] + grad_u_T[0][1]],
                [grad_u[1][0] + grad_u_T[1][0], grad_u[1][1] + grad_u_T[1][1]],
            ]
        )

        # Exact stress tensor (\sigma(\epsilon(u)))
        sigma = [
            [2 * mu * strain[0][0] + lam * trace_grad_u, 2 * mu * strain[0][1]],
            [2 * mu * strain[1][0], 2 * mu * strain[1][1] + lam * trace_grad_u],
        ]
        return sigma

    def evaluate_exact_force(
        self,
        sd: pp.Grid,
        time: float,
        sigma: sym.Expr,
        inds: np.ndarray,
        force_array: np.ndarray,
    ) -> np.ndarray:
        """Exact elastic force at the face centers for certain face indices.

        Important usage note: This method is typically called twice in the same
        simulation. The exact elastic force values may be different in the two parts of
        the subdomain. In the first call, the method fills half the `force_array`. The
        other half is filled in the second call. The filling of the array is done index
        wise, determined by `inds`. See evaluate_exact_force_split_domain() for example
        usage.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.
            sigma: Exact stress tensor.
            inds: The face indices which we are computing the force at.
            force_array: Either empty or semi empty array of shape (self.nd, sd.
                num_faces) which we are filling with the force values. This is done by
                the indices in `inds`.

        Returns:
            Array of containing the exact elastic force at the selected face centers for
            the given time.

        Notes:
            * The returned elastic force is _not_ given in PorePy's flattened vector
              format. Thus, it may be necessary to flatten it at a later point.
            * Recall that force = (stress dot_prod unit_normal) * face_area.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        fc = sd.face_centers[:, inds].squeeze()
        fn = sd.face_normals[:, inds].squeeze()

        # Lambdify expression
        sigma_total_fun = [
            [
                sym.lambdify((x, y, t), sigma[0][0], "numpy"),
                sym.lambdify((x, y, t), sigma[0][1], "numpy"),
            ],
            [
                sym.lambdify((x, y, t), sigma[1][0], "numpy"),
                sym.lambdify((x, y, t), sigma[1][1], "numpy"),
            ],
        ]

        # Face-centered elastic force
        force_total_fc: list[np.ndarray] = [
            # (sigma_xx * n_x + sigma_xy * n_y) * face_area
            sigma_total_fun[0][0](fc[0], fc[1], time) * fn[0]
            + sigma_total_fun[0][1](fc[0], fc[1], time) * fn[1],
            # (sigma_yx * n_x + sigma_yy * n_y) * face_area
            sigma_total_fun[1][0](fc[0], fc[1], time) * fn[0]
            + sigma_total_fun[1][1](fc[0], fc[1], time) * fn[1],
        ]

        # Insert values into force_array at the indices given by inds
        for i, force_component in enumerate(force_total_fc):
            # Update the appropriate row of force_array
            force_array[i, inds] = force_component

        return force_array

    def evaluate_exact_force_split_domain(self, sd: pp.Grid) -> np.ndarray:
        """Evaluate the exact force in the entire split domain.

        The domain is split in 2: One left and one right part, where the exact force may
        be different in each part of the domain. This method handles computing the exact
        force values for the entire domain, one region at a time.

        Parameters:
            sd: The subdomain grid where the forces are to be evaluated.

        Returns:
            A flattened array of the exact force values in the entire domain.

        """
        x, _, u_left, u_right = self.analytical_solution(lambdify=False)

        mu_lambda_values = {
            "left": (self.lmbda_left, self.mu_left),
            "right": (
                self.lmbda_right,
                self.mu_right,
            ),
        }

        sigma = {}
        for side, (lam, mu) in mu_lambda_values.items():
            u = u_left if side == "left" else u_right
            sigma[side] = self.exact_discontinuous_sigma(u, lam, mu, x)

        fc_x = sd.face_centers[0, :]

        empty_force_array = np.zeros((self.nd, sd.num_faces))

        inds_left = np.where(fc_x < self.discontinuity_location)
        inds_right = np.where(fc_x >= self.discontinuity_location)

        semi_full_force_array = self.evaluate_exact_force(
            sd, self.time_manager.time, sigma["left"], inds_left, empty_force_array
        )
        full_force_array = self.evaluate_exact_force(
            sd,
            self.time_manager.time,
            sigma["right"],
            inds_right,
            semi_full_force_array,
        )
        full_force_array: np.ndarray = np.asarray(full_force_array).ravel("F")
        return full_force_array

    def exact_fracture_traction_and_displacement_jump(
        self, sd: pp.Grid
    ) -> tuple[np.ndarray, np.ndarray]:
        """Computes exact fracture traction and displacement jump at the fracture."""
        if self.params.get("compressive_setup"):
            stiffness = self.solid.fracture_normal_stiffness
        else:
            stiffness = self.solid.fracture_tangential_stiffness

        # Find displacement jump instead
        L = self.discontinuity_location
        sd_frac = self.mdg.subdomains(dim=1)[0]
        L_vec = np.ones(sd_frac.num_cells) * L
        component = 0 if self.compressive_setup else 1

        left_solution, right_solution = self.analytical_solution()
        left_evaluated_mid = left_solution[component](L_vec, self.time_manager.time)
        right_evaluated_mid = right_solution[component](L_vec, self.time_manager.time)
        displacement_jump = right_evaluated_mid - left_evaluated_mid
        u = np.zeros((self.nd, sd_frac.num_cells))

        # As the displacement jump is determined with a coordinate system relative to
        # the fracture and not the global system, component 1 should be correct here for
        # compressive.
        component = 1 if self.compressive_setup else 0
        u[component, :] = displacement_jump

        # Return displacement jump and the displacement jump scaled with stiffness
        # (fracture traction)
        return u, u * stiffness

    def evaluate_mechanics_source(self, f: list, sd: pp.Grid, t: float) -> np.ndarray:
        vals = np.zeros((self.nd, sd.num_cells))
        return vals.ravel("F")


class Export:
    def data_to_export(self):
        """Export the analytical solution and compute errors at the final time step."""
        data = super().data_to_export()
        sd = self.mdg.subdomains(dim=2)[0]
        t = self.time_manager.time
        x = sd.cell_centers[0, :]

        L = self.discontinuity_location
        left_layer = x <= L
        right_layer = x > L

        values = np.zeros((self.nd, sd.num_cells))

        sol_left, sol_right = self.analytical_solution()

        component = 0 if self.compressive_setup else 1
        values[component][left_layer] += sol_left[component](x[left_layer], t)
        values[component][right_layer] += sol_right[component](x[right_layer], t)
        values = values.ravel("F")
        data.append((sd, "analytical_discontinuity", values))

        if self.time_manager.final_time_reached():
            self.compute_and_save_errors(
                filename=self.filename_path,
                filename_fracture=self.filename_path_fracture,
            )
        return data

    def compute_and_save_errors(self, filename: str, filename_fracture: str) -> None:
        """Compute and save errors between the analytical and numerical solutions.

        Parameters:
            filename: The path to the file where the errors are to be saved.
            filename_fracture: The path to the file where the fracture errors are to be
                saved.
        """
        sd = self.mdg.subdomains(dim=self.nd)[0]
        sd_frac = self.mdg.subdomains(dim=1)
        x = sd.cell_centers[0, :]
        L = self.discontinuity_location

        # Quantities in the rock bulk
        left_solution, right_solution = self.analytical_solution()
        left_layer = x <= L
        right_layer = x > L

        vals = np.zeros((self.nd, sd.num_cells))
        component = 0 if self.compressive_setup else 1
        vals[component, left_layer] = left_solution[component](
            x[left_layer], self.time_manager.time
        )
        vals[component, right_layer] = right_solution[component](
            x[right_layer], self.time_manager.time
        )

        displacement_ad = self.displacement([sd])
        u_approximate = self.equation_system.evaluate(displacement_ad)
        exact_displacement = vals.ravel("F")

        exact_force = self.evaluate_exact_force_split_domain(sd=sd)
        force_ad = self.stress([sd])
        approx_force = self.equation_system.evaluate(force_ad)
        error_u = ConvergenceAnalysis.lp_error(
            grid=sd,
            true_array=exact_displacement,
            approx_array=u_approximate,
            is_cc=True,
            relative=True,
        )
        error_t = ConvergenceAnalysis.lp_error(
            grid=sd,
            true_array=exact_force,
            approx_array=approx_force,
            is_cc=False,
            relative=True,
        )
        with open(filename, "a") as file:
            num_cells = sd.num_cells
            file.write(
                f"{num_cells}, {self.time_manager.time_index}, {error_u}, {error_t}\n"
            )

        # Quantities on the fracture
        displacement_jump_exact, fracture_traction_exact = (
            self.exact_fracture_traction_and_displacement_jump(sd=sd)
        )
        displacement_jump_exact = displacement_jump_exact.ravel("F")
        fracture_traction_exact = fracture_traction_exact.ravel("F")

        # Numerical fracture contact traction (scaled)
        t_frac = self.contact_traction(sd_frac)
        t_char = self.characteristic_contact_traction(sd_frac)
        t_scaled = t_frac * t_char
        fracture_traction_numerical = self.equation_system.evaluate(t_scaled)

        # Numerical displacement jump
        displacement_jump_numerical = self.equation_system.evaluate(
            self.displacement_jump(sd_frac)
        )

        error_u_frac = ConvergenceAnalysis.lp_error(
            grid=sd_frac[0],
            true_array=displacement_jump_exact,
            approx_array=displacement_jump_numerical,
            is_cc=True,
            relative=True,
        )

        error_t_frac = ConvergenceAnalysis.lp_error(
            grid=sd_frac[0],
            true_array=fracture_traction_exact,
            approx_array=fracture_traction_numerical,
            is_cc=True,
            relative=True,
        )

        with open(filename_fracture, "a") as file:
            num_cells = sd_frac[0].num_cells
            time_index = self.time_manager.time_index
            file.write(f"{num_cells}, {time_index}, {error_u_frac}, {error_t_frac}\n")


class SpringTypeLinearConvergenceSetup(
    IterativeSolverMixin,
    Geometry,
    MiscellaneousConstants,
    ReflectionTransmissionMethods,
    BoundaryAndInitialConditions,
    ExactExpressionsAndEvaluationMethods,
    Export,
    DynamicMomentumBalanceLinearSpringModel,
):
    """Model class for the spring-type fracture deformation convergence analyses."""
