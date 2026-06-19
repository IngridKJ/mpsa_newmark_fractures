import sys
from functools import partial
from typing import Union

import numpy as np
import porepy as pp
from porepy.models.momentum_balance import MomentumBalance

sys.path.append("../")
from porepy.numerics.fv import _fvutils
from porepy.viz.data_saving_model_mixin import FractureDeformationExporting

import time_derivatives
from utils import (acceleration_velocity_displacement, body_force_function, u_v_a_wrap)
from utils.constitutive_laws import BartonBandisSpringModel, LinearSpringModel


class NamesAndConstants:
    def _is_nonlinear_problem(self) -> bool:
        """All problem setups in this repository are linear."""
        return False

    @property
    def beta(self) -> float:
        """Newmark time discretization parameter, beta.

        Discretization parameter which somehow represents how the acceleration is
        averaged over the coarse of one time step. The value of beta = 0.25 corresponds
        to the "Average acceleration method".

        """
        return self.params.get("newmark_beta", 0.25)

    @property
    def gamma(self) -> float:
        """Newmark time discretization parameter, gamma.

        Discretization parameter which somehow represents the amount of numerical
        damping (positive, negative or zero). The value of gamma = 0.5 corresponds to
        no numerical damping.

        """
        return self.params.get("newmark_gamma", 0.5)

    @property
    def velocity_key(self) -> str:
        """Key/Name for the velocity variable/operator.

        Velocity is represented by a time dependent dense array with the name provided
        by this property, namely "velocity".

        """
        return "velocity"

    @property
    def acceleration_key(self) -> str:
        """Key/Name for acceleration variable/operator.

        Acceleration is represented by a time dependent dense array with the name
        provided by this property, namely "acceleration".

        """

        return "acceleration"

    @property
    def bc_values_mechanics_key(self) -> str:
        """Key for mechanical boundary conditions in the data dictionary."""
        return "bc_values_mechanics"

    def primary_wave_speed(self, is_scalar: bool = False) -> Union[float, np.ndarray]:
        """Primary wave speed.

        Speed of the compressive elastic waves.

        Parameters:
            is_scalar: Whether the primary wave speed should be scalar or not. Relevant
                for use with manufactured solutions which only supports a scalar wave
                speed for now.

        Returns:
            The value of the compressive elastic waves. Either scalar valued or the
            cell-center values, depending on the input parameter.

        """
        if not is_scalar:
            return np.sqrt((self.lambda_vector + 2 * self.mu_vector) / self.rho_vector)
        else:
            return np.sqrt(
                (
                    (self.solid.lame_lambda + 2 * self.solid.shear_modulus)
                    / self.solid.density
                )
            )

    def secondary_wave_speed(self, is_scalar: bool = False) -> Union[float, np.ndarray]:
        """Secondary wave speed.

        Speed of the shear elastic waves.

        Parameters:
            is_scalar: Whether the primary wave speed should be scalar or not. Relevant
                for use with manufactured solutions which only supports a scalar wave
                speed for now.

        Returns:
            The value of the shear elastic waves. Either scalar valued or the
            cell-center values, depending on the input parameter.

        """
        if not is_scalar:
            return np.sqrt(self.mu_vector / self.rho_vector)
        else:
            return np.sqrt(self.solid.shear_modulus / self.solid.density)

class BoundaryAndInitialConditions:
    @property
    def discrete_robin_weight_coefficient(self) -> float:
        """Additional coefficient for discrete Robin boundary conditions.

        After discretizing the time derivative in the absorbing boundary condition
        expressions, there might appear coefficients additional to those within the
        coefficient matrix. This model property assigns that coefficient.

        Returns:
            The Robin weight coefficient.

        """
        return 3 / 2

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Boundary condition type for the absorbing boundary condition model class.

        Assigns Robin boundaries to all subdomain boundaries by default. This also
        includes setting the Robin weight.

        Parameters:
            sd: The subdomain whose boundaries are to be set.

        Returns:
            The vectorial boundary condition operator for subdomain sd.

        """
        # Fetch boundary sides and assign type of boundary condition for the different
        # sides
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.north
            + bounds.south
            + bounds.east
            + bounds.west
            + bounds.bottom
            + bounds.top,
            "rob",
        )

        # Calling helper function for assigning the Robin weight
        self.assign_robin_weight(sd=sd, bc=bc)
        return bc

    def assign_robin_weight(
        self, sd: pp.Grid, bc: pp.BoundaryConditionVectorial
    ) -> None:
        """Assigns the Robin weight for Robin boundary conditions.

        Parameters:
            sd: The subdomain whose boundary conditions are to be defined.
            bc: The vectorial boundary condition object.

        """
        # Initiating the arrays for the Robin weight
        r_w = np.tile(np.eye(sd.dim), (1, sd.num_faces))
        value = np.reshape(r_w, (sd.dim, sd.dim, sd.num_faces), "F")

        # The Robin weight should only be assigned to subdomains of max dimension.
        if sd.dim != self.nd:
            bc.robin_weight = value
        else:
            # The coefficient matrix is constructed elsewhere, so we fetch it by making
            # this call.
            total_coefficient_matrix = self.total_coefficient_matrix(
                sd=sd,
            )

            # Discretizing the u_t term in the absorbing boundaries introduces an
            # additional weight to the u-term of the resulting Robin boundary condition
            # (see the method total_coefficient_matrix).
            # The next line includes that weight.
            total_coefficient_matrix *= self.discrete_robin_weight_coefficient

            # Fethcing all boundary faces for the domain and assigning the
            # total_coefficient_matrix to the boundary faces.
            boundary_faces = sd.get_boundary_faces()
            value[:, :, boundary_faces] *= total_coefficient_matrix.T

            # Finally setting the actual Robin weight.
            bc.robin_weight = value

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Method for assigning Robin boundary condition values (the right-hand side).

        Specifically, this method assigns the values corresponding to absorbing boundary
        conditions when we have used a second order approximation to u_t in:

            sigma * n + alpha * u_t = G

        See the method total_coefficient_matrix() for the expressions.

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

        # According to the expression for the absorbing boundaries, we have a
        # coefficient 2 in front of the values u_(n-1) and -0.5 in front of u_(n-2):
        displacement_values = 2 * displacement_values_0 - 0.5 * displacement_values_1

        # Transposing and reshaping displacement values to prepare for broadcasting
        displacement_values = displacement_values.reshape(
            boundary_grid.num_cells, self.nd, 1
        )

        # Assembling the vector representing the RHS of the Robin conditions
        total_coefficient_matrix = self.total_coefficient_matrix(sd=sd)
        robin_rhs = np.matmul(total_coefficient_matrix, displacement_values).squeeze(-1)
        robin_rhs *= sd.face_areas[boundary_faces][:, None]
        robin_rhs = robin_rhs.T
        return robin_rhs.ravel("F")

    def total_coefficient_matrix(self, sd: pp.Grid) -> np.ndarray:
        """Coefficient matrix for the absorbing boundary conditions.

        This method is used together with Robin boundary condition (absorbing boundary
        condition) assignment.

        The absorbing boundary conditions, in the continuous form, look like the
        following:

            sigma * n + D * u_t = 0,

        where u_t is the velocity and  D is a matrix containing material parameters
        and wave velocities.

        Approximate the time derivative by a second order backward difference:

            sigma * n + 3 / 2 * D_t * u_n = D_t * (2 * u_(n-1) - 0.5 * u_(n-2)),

        where D_t = 1/dt * D. Note the coefficient 3 / 2 (termed
        discrete_robin_weight_coefficient in the method assign_robin_weight)

        This method creates D_t, which is subsequently used both as a part of the weight
        in front of u_n and in the right hand side.

        Parameters:
            sd: The subdomain where the boundary conditions are assigned.

        Returns:
            A block array which is the sum of the normal and tangential component of the
            coefficient matrices.

        """
        # Fetching necessary grid related quantities
        boundary_cells = self.boundary_cells_of_grid
        boundary_faces = sd.get_boundary_faces()
        face_normals = sd.face_normals[:, boundary_faces][: self.nd]
        unitary_face_normals = face_normals / np.linalg.norm(
            face_normals, axis=0, keepdims=True
        )

        # This line of code does the same as the lines that are commented out
        # below:
        tensile_matrices = np.einsum(
            "ik,jk->kij", unitary_face_normals, unitary_face_normals
        )
        # tensile_matrices = np.array(
        #     [
        #         np.outer(normal_vector, normal_vector)
        #         for normal_vector in unitary_face_normals.T
        #     ]
        # )

        # Creating a block array of identity matrices. Subtracting the tensile matrix
        # block array from this provides us with the shear matrix block array.
        eye_block_array = np.tile(np.eye(self.nd), (len(tensile_matrices), 1, 1))
        shear_mat = eye_block_array - tensile_matrices

        # Scaling with the Robin weight value
        tensile_coeff = self.robin_weight_value(direction="tensile")[boundary_cells]
        shear_coeff = self.robin_weight_value(direction="shear")[boundary_cells]

        # Constructing the total coefficient matrix.
        tensile_matrix_with_coeff = tensile_matrices * tensile_coeff[:, None, None]
        shear_matrix_with_coeff = shear_mat * shear_coeff[:, None, None]
        total_coefficient_matrix = tensile_matrix_with_coeff + shear_matrix_with_coeff
        return total_coefficient_matrix

    def robin_weight_value(self, direction: str) -> float:
        """Robin weight value for the Robin (Absorbing) boundary conditions.

        The weight in the Robin (Absorbing) boundary conditions. The Robin weight is
        different for the tensile "direction" and the shear directions of a surface.
        Either shear or tensile Robin weight will be returned by this method. This
        depends on whether shear or tensile "direction" is chosen.

        Parameters:
            direction: Whether the boundary condition that uses the weight is the shear
                or tensile component of the displacement.

        Returns:
            The weight/coefficient for use in the Robin boundary conditions.

        """
        dt = self.time_manager.dt
        rho = self.rho_vector
        mu = self.mu_vector

        if direction == "shear":
            value = np.sqrt(rho * mu) * 1 / dt
        elif direction == "tensile":
            lmbda = self.lambda_vector
            value = np.sqrt(rho * (lmbda + 2 * mu)) * 1 / dt
        return value

    def boundary_displacement(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Method for reconstructing the boundary displacement.

        For now the only usage is within the "realm" of absorbing boundary conditions:
        we need the displacement values on the boundary at the previous time step.

        Note: This is for the pure mechanical problem. Modifications are needed when a
            coupling to fluid flow is introduced at some later point.

        Parameters:
            subdomains: List of subdomains. Should be of co-dimension 0.

        Returns:
            Ad operator representing the displacement on grid faces of the subdomains.

        """
        # Discretization
        discr = self.stress_discretization(subdomains)

        # Boundary conditions on external boundaries
        bc = self._combine_boundary_operators(  # type: ignore [call-arg]
            subdomains=subdomains,
            dirichlet_operator=self.displacement,
            neumann_operator=self.mechanical_stress,
            robin_operator=self.mechanical_stress,
            bc_type=self.bc_type_mechanics,
            dim=self.nd,
            name=self.bc_values_mechanics_key,
        )
        # Displacement
        displacement = self.displacement(subdomains)

        boundary_displacement = (
            discr.bound_displacement_cell() @ displacement
            + discr.bound_displacement_face() @ bc
        )

        boundary_displacement.set_name("boundary_displacement")
        return boundary_displacement

    def construct_and_save_boundary_displacement(
        self, boundary_grid: pp.BoundaryGrid
    ) -> None:
        """Construct and save boundary displacement for usage in boundary conditions.

        The absorbing boundary conditions require the previous displacement values on
        the domain boundary. This method makes sure that the displacement values two
        time steps back in time is stored on the appropriate time step solution index.
        Additionally it constructs the previous boundary displacement, and saves it to
        the current time step solution index (= 0).

        Paramters:
            boundary_grid: The grid that the displacement values are evaluated on.

        """
        data = self.mdg.boundary_grid_data(boundary_grid)
        name = "boundary_displacement_values"

        # The displacement value for the previous time step is constructed and the
        # one two time steps back in time is fetched from the dictionary.
        location = pp.TIME_STEP_SOLUTIONS
        pp.shift_solution_values(
            name=name,
            data=data,
            location=location,
            max_index=2,
        )

        sd = boundary_grid.parent
        displacement_boundary_operator = self.boundary_displacement([sd])
        displacement_values = self.equation_system.evaluate(
            displacement_boundary_operator
        )

        displacement_values_0 = boundary_grid.projection(self.nd) @ displacement_values
        pp.set_solution_values(
            name=name,
            values=displacement_values_0,
            data=data,
            time_step_index=0,
        )


class InitialConditionsDynamicMomentumBalance:
    def set_initial_values_primary_variables(self) -> None:
        super().set_initial_values_primary_variables()

        # Velocity and acceleration is only defined on grids with ambient dimension.
        for sd, data in self.mdg.subdomains(return_data=True, dim=self.nd):
            ic_values_velocity = self.ic_values_velocity(sd=sd)
            ic_values_acceleration = self.ic_values_acceleration(sd=sd)

            pp.set_solution_values(
                name=self.velocity_key,
                values=ic_values_velocity,
                data=data,
                time_step_index=0,
                iterate_index=0,
            )

            pp.set_solution_values(
                name=self.acceleration_key,
                values=ic_values_acceleration,
                data=data,
                time_step_index=0,
                iterate_index=0,
            )

    def ic_values_displacement(self, sd: pp.Grid) -> np.ndarray:
        """Cell-centered initial displacement values.

        Parameters:
            dofs: Number of degrees of freedom (typically cell number in the grid where
                the initial values are defined).

        Returns:
            An array with the initial displacement values.

        """
        t = self.time_manager.time

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]

        vals = np.zeros((self.nd, sd.num_cells))

        if self.nd == 2:
            displacement_function = u_v_a_wrap(self)

            vals[0] = displacement_function[0](x, y, t)
            vals[1] = displacement_function[1](x, y, t)

        elif self.nd == 3:
            z = sd.cell_centers[2, :]

            displacement_function = u_v_a_wrap(self, is_2D=False)

            vals[0] = displacement_function[0](x, y, z, t)
            vals[1] = displacement_function[1](x, y, z, t)
            vals[2] = displacement_function[2](x, y, z, t)

        return vals.ravel("F")

    def ic_values_velocity(self, sd: pp.Grid) -> np.ndarray:
        """Cell-centered initial velocity values.

        Parameters:
            dofs: Number of degrees of freedom (typically cell number in the grid where
                the initial values are defined).

        Returns:
            An array with the initial velocity values.

        """
        t = self.time_manager.time

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]

        vals = np.zeros((self.nd, sd.num_cells))

        if self.nd == 2:
            velocity_function = u_v_a_wrap(self, return_dt=True)

            vals[0] = velocity_function[0](x, y, t)
            vals[1] = velocity_function[1](x, y, t)

        elif self.nd == 3:
            z = sd.cell_centers[2, :]

            velocity_function = u_v_a_wrap(self, is_2D=False, return_dt=True)

            vals[0] = velocity_function[0](x, y, z, t)
            vals[1] = velocity_function[1](x, y, z, t)
            vals[2] = velocity_function[2](x, y, z, t)

        return vals.ravel("F")

    def ic_values_acceleration(self, sd: pp.Grid) -> np.ndarray:
        """Cell-centered initial acceleration values.

        Parameters:
            dofs: Number of degrees of freedom (typically cell number in the grid where
                the initial values are defined).

        Returns:
            An array with the initial acceleration values.

        """
        t = self.time_manager.time

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]

        vals = np.zeros((self.nd, sd.num_cells))

        if self.nd == 2:
            acceleration_function = u_v_a_wrap(self, return_ddt=True)

            vals[0] = acceleration_function[0](x, y, t)
            vals[1] = acceleration_function[1](x, y, t)

        elif self.nd == 3:
            z = sd.cell_centers[2, :]

            acceleration_function = u_v_a_wrap(self, is_2D=False, return_ddt=True)

            vals[0] = acceleration_function[0](x, y, z, t)
            vals[1] = acceleration_function[1](x, y, z, t)
            vals[2] = acceleration_function[2](x, y, z, t)

        return vals.ravel("F")

    def ic_values_bc(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Sets the initial bc values for 0th and -1st time step in the data dictionary.

        We need initial bc values two time steps back in time. This method sets the
        values for 0th and -1st time step into the data dictionary. It also returns the
        0th time step values, which is needed when the bc_robin_values is called the
        first time (at initialization).

        Parameters:
            bg: Boundary grid whose boundary displacement value is to be set.

        Returns:
            An array with the initial displacement boundary values.

        """
        vals_0 = self.ic_values_bc_0(bg=bg)
        vals_1 = self.ic_values_bc_1(bg=bg)

        data = self.mdg.boundary_grid_data(bg)

        name = "boundary_displacement_values"
        # The values for the 0th and -1th time step are to be stored
        pp.set_solution_values(
            name=name,
            values=vals_1,
            data=data,
            time_step_index=1,
        )
        pp.set_solution_values(
            name=name,
            values=vals_0,
            data=data,
            time_step_index=0,
        )
        return vals_0

    def ic_values_bc_0(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Initial boundary displacement values corresponding to time step 0."""
        return np.zeros((self.nd, bg.num_cells)).ravel("F")

    def ic_values_bc_1(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Initial boundary displacement values corresponding to time step -1."""
        return np.zeros((self.nd, bg.num_cells)).ravel("F")


class DynamicMomentumBalanceEquations:
    def momentum_balance_equation(self, subdomains: list[pp.Grid]):
        """Dynamic momentum balance equation in the rock matrix.

        Parameters:
            subdomains: List of subdomains where the force balance is defined.

        Returns:
            Operator for the force balance equation in the matrix.

        """
        inertia_mass = self.inertia(subdomains)
        stress = pp.ad.Scalar(-1) * self.stress(subdomains)
        body_force = self.body_force(subdomains)

        equation = self.balance_equation(
            subdomains, inertia_mass, stress, body_force, dim=self.nd
        )
        equation.set_name("momentum_balance_equation")
        return equation

    def inertia(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Inertia mass in the elastic wave equation/dynamic momentum balance equation.

        The elastic wave equation contains a term on the form M * u_tt (that is, M *
        acceleration term/inertia term). This method represents an operator for M.

        Parameters:
            subdomains: List of subdomains where the inertia mass is defined.

        Returns:
            Operator for the inertia mass.

        """
        # In case of a problem with a heterogeneous rock density, the mass density term
        # must be generalised to be cell wise instead of a scalar.
        mass_density = pp.ad.DenseArray(np.repeat(self.rho_vector, self.nd))
        mass = self.volume_integral(mass_density, subdomains, dim=self.nd)
        mass.set_name("inertia_mass")
        return mass

    def balance_equation(
        self,
        subdomains: list[pp.Grid],
        inertia_mass: pp.ad.Operator,
        surface_term: pp.ad.Operator,
        source: pp.ad.Operator,
        dim: int,
    ) -> pp.ad.Operator:
        """Balance equation that combines an acceleration and surface term.

        The balance equation, namely the elastic wave equation, is given by
            d_tt(displacement) + div(surface_term) - source = 0.

        Parameters:
            subdomains: List of subdomains where the balance equation is defined.
            inertia_mass: Operator for the cell-wise mass of the acceleration term,
                integrated over the cells of the subdomains.
            surface_term: Operator for the surface term (e.g. flux, stress), integrated
                over the faces of the subdomains.
            source: Operator for the source term, integrated over the cells of the
                subdomains.
            dim: Spatial dimension of the balance equation.

        Returns:
            Operator for the balance equation.

        """
        div = pp.ad.Divergence(subdomains, dim=dim)

        # Fetch the necessary operators for creating the acceleration operator
        op = self.displacement(subdomains)
        dt_op = self.velocity_time_dep_array(subdomains)
        ddt_op = self.acceleration_time_dep_array(subdomains)

        # Create the acceleration operator:
        inertia_term = time_derivatives.inertia_term(
            model=self,
            op=op,
            dt_op=dt_op,
            ddt_op=ddt_op,
            time_step=self.ad_time_step,
        )

        return inertia_mass * inertia_term + div @ surface_term - source


class SolutionStrategyDynamicMomentumBalance:
    def prepare_simulation(self) -> None:
        """Run at the start of simulation. Used for initialization etc.

        This method overrides the original prepare_simulation. The reason for this is
        that we need to create the attributes boundary_cells_of_grid, lambda_vector and
        mu_vector. All of which are needed in the call to initial_condition as well as
        set_equations.

        """
        # Set the material and geometry of the problem. The geometry method must be
        # implemented in a ModelGeometry class.
        self.set_materials()
        self.set_geometry()

        self.set_vector_valued_mu_lambda_rho()
        # Exporter initialization must be done after grid creation,
        # but prior to data initialization.
        self.set_nonlinear_solver_statistics()
        self.initialize_data_saving()

        # Set variables, constitutive relations, discretizations and equations.
        # Order of operations is important here.
        self.set_equation_system_manager()
        self.create_variables()

        # After fluid and variables are defined, we can define the secondary quantities
        # like fluid properties (which depend on variables). Creating fluid and
        # variables before defining secondary thermodynamic properties is critical in
        # the case where properties depend on some fractions. since the callables for
        # secondary variables are dynamically created during create_variables, as
        # opposed to e.g. pressure or temperature.
        self.assign_thermodynamic_properties_to_phases()
        self.initial_condition()
        self.initialize_previous_iterate_and_time_step_values()

        # Initialize time dependent ad arrays, including those for boundary values.
        self.update_time_dependent_ad_arrays()
        self.reset_state_from_file()
        self.set_equations()

        self.update_discretization_parameters()
        self.discretize()
        self._initialize_linear_solver()
        self.set_nonlinear_discretizations()

        # # Sets initial acceleration. Beware that this will override the values set by
        # # ic_values_acceleration. Some care should be taken in this regard sometime.
        # if self._is_nonlinear_problem():
        #     self.set_post_processed_initial_acceleration()

        # Export initial condition
        self.save_data_time_step()

    def set_vector_valued_mu_lambda_rho(self) -> None:
        """Instantiate boundary cell information and assign vector valued mu and lambda.

        Allow mu and lambda (Lamé parameters) to be assigned as cell wise values instead
        of only scalars.

        Additionally, creating the coefficient matrix for the absorbing
        boundaries requires the attribute boundary_cells_of_grid to distinguish which
        cells are bordering the domain boundary. boundary_cells_of_grid is therefore
        defined as an attribute of the model class setup here.

        """
        sd = self.mdg.subdomains(dim=self.nd)[0]
        boundary_faces = sd.get_boundary_faces()
        self.boundary_cells_of_grid = sd.signs_and_cells_of_boundary_faces(
            faces=boundary_faces
        )[1]
        self.vector_valued_mu_lambda_rho()

    def update_discretization_parameters(self) -> None:
        """Set discretization parameters for the simulation.

        Sets eta = 1/3 on all faces if it is a simplex grid. Default is to have 0 on the
        boundaries, but this causes divergence of the solution. 1/3 for all subfaces in
        the grid fixes this issue.

        """
        super().update_discretization_parameters()
        if self.params["grid_type"] == "simplex":
            num_subfaces = 0
            for sd, data in self.mdg.subdomains(return_data=True, dim=self.nd):
                subcell_topology = _fvutils.SubcellTopology(sd)
                num_subfaces += subcell_topology.num_subfno
                eta_values = np.ones(num_subfaces) * 1 / 3
                if sd.dim == self.nd:
                    pp.initialize_data(
                        data,
                        self.stress_keyword,
                        {
                            "mpsa_eta": eta_values,
                        },
                    )

    def velocity_time_dep_array(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.TimeDependentDenseArray:
        """Time dependent dense array for the velocity.

        Creates a time dependent dense array to represent the velocity, which is needed
        for the Newmark time discretization.

        Parameters:
            subdomains: List of subdomains on which to define the velocity.

        Returns:
            Operator representation of the acceleration.

        """
        return pp.ad.TimeDependentDenseArray(self.velocity_key, subdomains)

    def acceleration_time_dep_array(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.TimeDependentDenseArray:
        """Time dependent dense array for the acceleration.

        Creates a time dependent dense array to represent the acceleration, which is
        needed for the Newmark time discretization.

        Parameters:
            subdomains: List of subdomains on which to define the acceleration.

        Returns:
            Operator representation of the acceleration.

        """
        return pp.ad.TimeDependentDenseArray(self.acceleration_key, subdomains)

    def velocity_values(self, subdomain: pp.Grid) -> np.ndarray:
        """Update the velocity values at the end of each time step.

        The velocity values are updated once the system is solved for the cell-centered
        displacements (at the end of a time step). The values are updated by using the
        Newmark formula for velocity (see e.g. Dynamics of Structures by A. K. Chopra
        (pp. 175-176, 2014)).

        Parameters:
            subdomain: The subdomain the velocity is defined on.

        Returns:
            An array with the new velocity values.

        """
        data = self.mdg.subdomain_data(subdomain)
        dt = self.time_manager.dt

        beta = self.beta
        gamma = self.gamma

        (
            a_previous,
            v_previous,
            u_previous,
            u_current,
        ) = acceleration_velocity_displacement(model=self, data=data)

        v = (
            v_previous * (1 - gamma / beta)
            + a_previous * (1 - gamma - (gamma * (1 - 2 * beta)) / (2 * beta)) * dt
            + (u_current - u_previous) * gamma / (beta * dt)
        )
        return v

    def acceleration_values(self, subdomain: pp.Grid) -> np.ndarray:
        """Update the acceleration values at the end of each time step.

        See the method velocity_values for more extensive documentation.

        Parameters:
            subdomain: The subdomain the acceleration is defined on.

        Returns:
            An array with the new acceleration values.

        """
        data = self.mdg.subdomain_data(subdomain)
        dt = self.time_manager.dt

        beta = self.beta

        (
            a_previous,
            v_previous,
            u_previous,
            u_current,
        ) = acceleration_velocity_displacement(model=self, data=data)

        a = (
            (u_current - u_previous) / (dt**2 * beta)
            - v_previous / (dt * beta)
            - a_previous * (1 - 2 * beta) / (2 * beta)
        )
        return a

    def update_velocity_acceleration_time_dependent_ad_arrays(self) -> None:
        """Update the time dependent ad arrays for the velocity and acceleration.

        NB: Docs will be outdated after iterative adaptations are made.
        The new velocity and acceleration values (the value at the end of each time
        step) are set into the data dictionary.

        """
        sd, data = self.mdg.subdomains(return_data=True, dim=self.nd)[0]

        vals_acceleration = self.acceleration_values(sd)
        vals_velocity = self.velocity_values(sd)

        pp.set_solution_values(
            name=self.velocity_key,
            values=vals_velocity,
            data=data,
            time_step_index=0,
            iterate_index=0,
        )
        pp.set_solution_values(
            name=self.acceleration_key,
            values=vals_acceleration,
            data=data,
            time_step_index=0,
            iterate_index=0,
        )

    def update_time_step_solution(self) -> None:
        """Method to be called after every time step.

        The method update_velocity_acceleration_time_dependent_ad_arrays needs to be
        called at the end of each time step. This is not within PorePy itself, and
        therefore this method is overriding the default update_time_step_solution
        method.

        """
        self.update_velocity_acceleration_time_dependent_ad_arrays()
        solution = self.equation_system.get_variable_values(iterate_index=0)

        self.equation_system.shift_time_step_values()
        self.equation_system.set_variable_values(
            values=solution, time_step_index=0, additive=False
        )
        if self.time_manager.time_index >= 1:
            bg = self.mdg.boundaries(dim=self.nd - 1)[0]
            self.construct_and_save_boundary_displacement(boundary_grid=bg)


class ConstitutiveLawsDynamicMomentumBalance:
    def vector_valued_mu_lambda_rho(self) -> None:
        """Vector representation of mu and lambda.

        Cell-wise representation of the mu and lambda quantities in the rock matrix.
        Both are saved as an attribute of the model class setup.

        """
        subdomain = self.mdg.subdomains(dim=self.nd)[0]

        self.lambda_vector = self.solid.lame_lambda * np.ones(subdomain.num_cells)
        self.mu_vector = self.solid.shear_modulus * np.ones(subdomain.num_cells)
        self.rho_vector = self.solid.density * np.ones(subdomain.num_cells)

    def stiffness_tensor(self, subdomain: pp.Grid) -> pp.FourthOrderTensor:
        """Stiffness tensor [Pa].

        Overriding the stiffness_tensor method to accomodate vector valued mu and
        lambda. Vector valued mu and lambda is treated as default in the model class for
        MPSA-Newmark with absorbing boundaries.

        Parameters:
            subdomain: Subdomain where the stiffness tensor is defined.

        Returns:
            Cell-wise stiffness tensor in SI units.

        """
        return pp.FourthOrderTensor(self.mu_vector, self.lambda_vector)


class TimeDependentSourceTerm:
    def body_force(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Time dependent dense array for the body force term.

        Creates a time dependent dense array to represent the body force/source.

        Parameters:
            subdomains: List of subdomains on which to define the body force.

        Returns:
            Operator representation of the body force.

        """
        external_sources = pp.ad.TimeDependentDenseArray(
            name="source_mechanics",
            domains=subdomains,
        )
        return external_sources

    def before_nonlinear_loop(self) -> None:
        """Update the time dependent mechanics source before every nonlinear loop.

        The default behaviour is to save source values which correspond to a source
        which is constructed from a known analytical reference solution. In the case of
        no analytical reference solution being chosen (this is done in the params
        dictionary), the source is just zero.

        """
        super().before_nonlinear_loop()

        sd = self.mdg.subdomains(dim=self.nd)[0]
        data = self.mdg.subdomain_data(sd)
        t = self.time_manager.time

        # Mechanics source
        if self.nd == 2:
            mechanics_source_function = body_force_function(self)
        elif self.nd == 3:
            mechanics_source_function = body_force_function(self, is_2D=False)

        mechanics_source_values = self.evaluate_mechanics_source(
            f=mechanics_source_function, sd=sd, t=t
        )
        pp.set_solution_values(
            name="source_mechanics",
            values=mechanics_source_values,
            data=data,
            iterate_index=0,
        )

    def evaluate_mechanics_source(self, f: list, sd: pp.Grid, t: float) -> np.ndarray:
        """Computes the values for the body force.

        The method computes the source values returned by the source value function (f)
        integrated over the cell. The function is evaluated at the cell centers.

        Parameters:
            f: Function expression for the source term. It depends on time and space for
                the source term. It is represented as a list, where the first list
                component corresponds to the first vector component of the source,
                second list component corresponds to the second vector component, and so
                on.
            sd: Subdomain where the source term is defined.
            t: Current time in the time-stepping.

        Returns:
            An array of source values.

        """
        cell_volume = sd.cell_volumes
        vals = np.zeros((self.nd, sd.num_cells))

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]

        if self.nd == 2:
            x_val = f[0](x, y, t)
            y_val = f[1](x, y, t)

        elif self.nd == 3:
            z = sd.cell_centers[2, :]

            x_val = f[0](x, y, z, t)
            y_val = f[1](x, y, z, t)
            z_val = f[2](x, y, z, t)

            vals[2] = z_val * cell_volume

        vals[0] = x_val * cell_volume
        vals[1] = y_val * cell_volume
        return vals.ravel("F")


class DynamicMomentumBalanceABC(
    NamesAndConstants,
    BoundaryAndInitialConditions,
    InitialConditionsDynamicMomentumBalance,
    DynamicMomentumBalanceEquations,
    ConstitutiveLawsDynamicMomentumBalance,
    TimeDependentSourceTerm,
    SolutionStrategyDynamicMomentumBalance,
    MomentumBalance,
):
    """Model class setup for the dynamic momentum balance with absorbing boundaries."""

    def data_to_export(self):
        """"""
        data = super().data_to_export()

        sd = self.mdg.subdomains(dim=self.nd)[0]
        acc = self.acceleration_time_dep_array([sd])
        acceleration_value = acc.value(self.equation_system)
        vel = self.velocity_time_dep_array([sd])
        velocity_value = vel.value(self.equation_system)

        data.append((sd, "acceleration", acceleration_value))
        data.append((sd, "velocity", velocity_value))
        return data


class DynamicMomentumBalanceABCNonlinear(
    FractureDeformationExporting, DynamicMomentumBalanceABC
):
    """Model class setup for the dynamic momentum balance with absorbing boundaries for
    nonlinear problems.

    """

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        bc = super().bc_type_mechanics(sd=sd)
        bc.internal_to_dirichlet(sd)
        return bc

    def _is_nonlinear_problem(self) -> bool:
        """Asserting problem is nonlinear."""
        return True

    @property
    def velocity_jump_key(self) -> str:
        """Key/Name for the velocity jump variable/operator.

        Velocity jump is represented by a time dependent dense array with the name
        provided by this property, namely "velocity_jump".

        """
        return "velocity_jump"

    @property
    def acceleration_jump_key(self) -> str:
        """Key/Name for acceleration jump variable/operator.

        Acceleration jump is represented by a time dependent dense array with the name
        provided by this property, namely "acceleration_jump".

        """
        return "acceleration_jump"

    def set_initial_values_primary_variables(self) -> None:
        super().set_initial_values_primary_variables()

        # Velocity and acceleration jumps are only defined on fracture grids.
        for sd, data in self.mdg.subdomains(return_data=True, dim=self.nd - 1):
            ic_values_velocity_jump = self.ic_values_velocity_jump(sd=sd)
            ic_values_acceleration_jump = self.ic_values_acceleration_jump(sd=sd)

            pp.set_solution_values(
                name=self.velocity_jump_key,
                values=ic_values_velocity_jump,
                data=data,
                time_step_index=0,
                iterate_index=0,
            )

            pp.set_solution_values(
                name=self.acceleration_jump_key,
                values=ic_values_acceleration_jump,
                data=data,
                time_step_index=0,
                iterate_index=0,
            )

    def ic_values_velocity_jump(self, sd: pp.Grid) -> np.ndarray:
        """Initial velocity jump values."""
        return np.zeros(sd.num_cells * sd.dim)

    def ic_values_acceleration_jump(self, sd: pp.Grid) -> np.ndarray:
        """Initial acceleration jump values."""
        return np.zeros(sd.num_cells * sd.dim)

    def velocity_jump_time_dep_array(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.TimeDependentDenseArray:
        """Time dependent dense array for the velocity jump."""
        return pp.ad.TimeDependentDenseArray(self.velocity_jump_key, subdomains)

    def acceleration_jump_time_dep_array(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.TimeDependentDenseArray:
        """Time dependent dense array for the acceleration jump."""
        return pp.ad.TimeDependentDenseArray(self.acceleration_jump_key, subdomains)

    def tangential_velocity_jump(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Tangential component of the velocity jump.

        Constructs and returns the tangential velocity jump. The tangential component(s)
        of the velocity jump is the slip and is needed for the Newmark
        time-discretization.

        Parameters:
            subdomains: List of fracture subdomains where the velocity jump is defined.

        Returns:
            An ad operator representing the tangential velocity jump. It is 1 component
            in 2D and 2 components in 3D.

        """
        nd_vec_to_tangential = self.tangential_component(subdomains)
        displacement_jump_t = nd_vec_to_tangential @ self.displacement_jump(
            subdomains=subdomains
        )

        v_jump = self.velocity_jump_time_dep_array(subdomains=subdomains)
        a_jump = self.acceleration_jump_time_dep_array(subdomains=subdomains)

        velocity_jump_t = (
            pp.ad.Scalar(1 - self.gamma / self.beta) * v_jump.previous_timestep()
            + self.ad_time_step
            * pp.ad.Scalar(1 - self.gamma / (2 * self.beta))
            * a_jump.previous_timestep()
            + pp.ad.Scalar(self.gamma / self.beta)
            / self.ad_time_step
            * (displacement_jump_t - displacement_jump_t.previous_timestep())
        )
        return velocity_jump_t

    def tangential_acceleration_jump(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Tangential component of the acceleration jump.

        Constructs and returns the tangential acceleration jump. The tangential
        component(s) of the acceleration jump is needed for the Newmark
        time-discretization.

        Parameters:
            subdomains: List of fracture subdomains where the acceleration jump is
            defined.

        Returns:
            An ad operator representing the tangential acceleration jump. It is 1
            component in 2D and 2 components in 3D.

        """
        nd_vec_to_tangential = self.tangential_component(subdomains)
        displacement_jump_t = nd_vec_to_tangential @ self.displacement_jump(
            subdomains=subdomains
        )

        v_jump = self.velocity_jump_time_dep_array(subdomains=subdomains)
        a_jump = self.acceleration_jump_time_dep_array(subdomains=subdomains)

        acceleration_jump_t = (
            pp.ad.Scalar(1)
            / (pp.ad.Scalar(self.beta) * (self.ad_time_step * self.ad_time_step))
            * (
                displacement_jump_t
                - displacement_jump_t.previous_timestep()
                - self.ad_time_step * v_jump.previous_timestep()
                - pp.ad.Scalar(1 - 2 * self.beta)
                * (self.ad_time_step * self.ad_time_step)
                / pp.ad.Scalar(2)
                * a_jump.previous_timestep()
            )
        )
        return acceleration_jump_t

    def update_velocity_and_acceleration_jumps(self) -> None:
        """Update the velocity and acceleration jumps."""
        for sd, data in self.mdg.subdomains(return_data=True, dim=self.nd - 1):
            velocity_jump_operator = self.tangential_velocity_jump(subdomains=[sd])
            acceleration_jump_operator = self.tangential_acceleration_jump(
                subdomains=[sd]
            )
            velocity_jump = self.equation_system.evaluate(velocity_jump_operator)
            acceleration_jump = self.equation_system.evaluate(
                acceleration_jump_operator
            )

            pp.set_solution_values(
                name=self.velocity_jump_key,
                values=velocity_jump,
                data=data,
                time_step_index=0,
            )
            pp.set_solution_values(
                name=self.acceleration_jump_key,
                values=acceleration_jump,
                data=data,
                time_step_index=0,
            )

    def update_time_step_solution(self) -> None:
        """Shifts solutions and sets the most recent solution.

        The method update_velocity_acceleration_time_dependent_ad_arrays needs to be
        called at the end of each time step. This is not within PorePy itself, and
        therefore this method is overriding the default update_time_step_solution
        method. Additionally, the velocity and acceleration jumps are updated.

        """
        self.update_velocity_acceleration_time_dependent_ad_arrays()
        self.update_velocity_and_acceleration_jumps()

        solution = self.equation_system.get_variable_values(iterate_index=0)

        self.equation_system.shift_time_step_values()
        self.equation_system.set_variable_values(
            values=solution, time_step_index=0, additive=False
        )
        if self.time_manager.time_index >= 1:
            bg = self.mdg.boundaries(dim=self.nd - 1)[0]
            self.construct_and_save_boundary_displacement(boundary_grid=bg)

    def tangential_fracture_deformation_equation(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Contact mechanics equation for the tangential constraints.

        The equation is dimensionless, as we use nondimensionalized contact traction.
        The function reads
        .. math::
            C_t = max(b_p, ||T_t+c_t v_t||) T_t - max(0, b_p) (T_t+c_t v_t)

        with `v` being velocity jump, `t` denoting tangential component and `b_p` the
        friction bound.

        For `b_p = 0`, the equation `C_t = 0` does not in itself imply `T_t = 0`, which
        is what the contact conditions require. The case is handled through the use of a
        characteristic function.

        NOTE: This method is largely copy pasted from PorePy. The change is mainly that
        the increment in u is represented by the tangential velocity jump.

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            complementary_eq: Contact mechanics equation for the tangential constraints.

        """
        # The lines below is an implementation of equations (25) and (27) in the paper
        #
        # Berge et al. (2020): Finite volume discretization for poroelastic media with
        #   fractures modeled by contact mechanics (IJNME, DOI: 10.1002/nme.6238). The
        #
        # Note that:
        #  - We do not directly implement the matrix elements of the contact traction
        #    as are derived by Berge in their equations (28)-(32). Instead, we directly
        #    implement the complimentarity function, and let the AD framework take care
        #    of the derivatives.
        #  - Related to the previous point, we do not implement the regularization that
        #    is discussed in Section 3.2.1 of the paper.

        # Basis vector combinations
        num_cells = sum([sd.num_cells for sd in subdomains])
        # Mapping from a full vector to the tangential component
        nd_vec_to_tangential = self.tangential_component(subdomains)

        # Basis vectors for the tangential components. This is a list of Ad matrices,
        # each of which represents a cell-wise basis vector which is non-zero in one
        # dimension (and this is known to be in the tangential plane of the subdomains).
        tangential_basis = self.basis(subdomains, dim=self.nd - 1)

        # To map a scalar to the tangential plane, we need to sum the basis vectors. The
        # individual basis vectors can be represented as projection matrices of shape
        # (Nc * (self.nd - 1), Nc), where Nc is the total number of cells in the
        # subdomain. The matrix representation of the sum has the same shape, but the
        # row corresponding to each cell will be non-zero in all rows corresponding to
        # the tangential basis vectors of this cell.
        scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)

        # Variables: The tangential component of the contact traction and the plastic
        # displacement jump.
        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)

        # In the case of the dynamic problem, we use an actual velocity in the
        # tangential fracture deformation equation instead of a displacement increment.
        # Default behaviour is to use the tangential velocity jump instead of just the
        # tangential displacement jump:
        u_t_increment: pp.ad.Operator = (
            self.tangential_velocity_jump(subdomains=subdomains) * self.ad_time_step
        )

        # Vectors needed to express the governing equations
        ones_frac = pp.ad.DenseArray(np.ones(num_cells * (self.nd - 1)))
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells))

        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        # The numerical constant is used to loosen the sensitivity in the transition
        # between sticking and sliding. Expanding using only left multiplication to with
        # scalar_to_tangential does not work for an array, unlike the operators below.
        # Arrays need right multiplication as well.
        c_num_as_scalar = self.contact_mechanics_numerical_constant(subdomains)

        # The numerical parameter is a cell-wise scalar, or a single scalar common for
        # all cells. In both cases, it must be extended to a vector quantity to be used
        # in the equation (multiplied from the right). Do this by multiplying with the
        # sum of the tangential basis vectors. Then take a Hadamard product with the
        # tangential displacement jump and add to the tangential component of the
        # contact traction to arrive at the expression that enters the equation.
        tangential_sum = t_t + (scalar_to_tangential @ c_num_as_scalar) * u_t_increment

        norm_tangential_sum = f_norm(tangential_sum)
        norm_tangential_sum.set_name("norm_tangential")

        b_p = f_max(self.friction_bound(subdomains), zeros_frac)
        b_p.set_name("bp")

        bp_tang = (scalar_to_tangential @ b_p) * tangential_sum

        maxbp_abs = scalar_to_tangential @ f_max(b_p, norm_tangential_sum)

        # The characteristic function below reads "1 if (abs(b_p) < tol) else 0".
        # With the active set method, the performance of the Newton solver is sensitive
        # to changes in state between sticking and sliding. To reduce the sensitivity to
        # round-off errors, we use a tolerance to allow for slight inaccuracies before
        # switching between the two cases. The tolerance is a numerical method parameter
        # and can be tailored.
        characteristic = self.contact_mechanics_open_state_characteristic(subdomains)

        # Compose the equation itself. The last term handles the case bound=0, in which
        # case t_t = 0 cannot be deduced from the standard version of the complementary
        # function (i.e. without the characteristic function). Filter out the other
        # terms in this case to improve convergence
        equation: pp.ad.Operator = (ones_frac - characteristic) * (
            bp_tang - maxbp_abs * t_t
        ) + characteristic * t_t
        equation.set_name("tangential_fracture_deformation_equation")
        return equation

    def shear_dilation_gap(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Shear dilation [m].

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise shear dilation.

        """
        angle: pp.ad.Operator = self.dilation_angle(subdomains)
        f_norm = pp.ad.Function(
            partial(pp.ad.functions.l2_norm, self.nd - 1), "norm_function"
        )
        f_tan = pp.ad.Function(pp.ad.functions.tan, "tan_function")

        # This was plastic displacement jump. Now it uses the full displacement jump.
        shear_dilation: pp.ad.Operator = f_tan(angle) * f_norm(
            self.tangential_component(subdomains) @ self.displacement_jump(subdomains)
        )

        shear_dilation.set_name("shear_dilation")
        return shear_dilation

    def data_to_export(self) -> list:
        """Return data to be exported.

        Adds the following quantities to the data to be exported:
            * Fracture opening
            * Velocity jump
            * Acceleration jump

        """
        data = super().data_to_export()
        for dim in range(self.nd + 1):
            for sd, d in self.mdg.subdomains(dim=dim, return_data=True):
                # Fractures live on dimension nd-1
                if dim == self.nd - 1:
                    # Fracture opening. First we need displacement jump:
                    displacement_jump = self.evaluate_and_scale(
                        [sd], "displacement_jump", "m"
                    )
                    # Reshape to (nd, num_cells)
                    dj_reshaped = displacement_jump.reshape(
                        (self.nd, sd.num_cells), order="F"
                    )

                    # Normal component is index self.nd - 1
                    gap = self.evaluate_and_scale([sd], "fracture_gap", "m")

                    fracture_opening = dj_reshaped[self.nd - 1, :] - gap
                    data.append((sd, "fracture_opening", fracture_opening))

                    # Velocity jump
                    velocity_jump_values = pp.get_solution_values(
                        name=self.velocity_jump_key,
                        data=d,
                        time_step_index=0,
                    )
                    data.append((sd, "velocity_jump", velocity_jump_values))

                    # Acceleration jump
                    acceleration_jump_values = pp.get_solution_values(
                        name=self.acceleration_jump_key,
                        data=d,
                        time_step_index=0,
                    )
                    data.append((sd, "acceleration_jump", acceleration_jump_values))
        return data

    def set_post_processed_initial_acceleration(self) -> None:
        """Computes and sets the initial acceleration field.

        This method calculates the initial acceleration based on the elastic wave
        equation. The elastic wave equation is described as:

        .. math::
            M * u_tt + div(sigma) = source

        where :math:`M` is the mass matrix, :math:`u_tt` is the acceleration, and
        :math:`sigma` is the stress tensor.

        Solving this equation for the acceleration gives: .. math::
            u_tt = -M^{-1} * div(sigma) + source

        And performing this calculation in the beginning of the simulation, after the
        initial displacement, boundary conditions, equations, etc., have been set,
        provides us with the initial acceleration corresponding to the problem. Note
        that source term is neglected here, so remember to set it != 0 if a non-zero
        source term is used.

        The computed acceleration is stored in the solution data structure.

        Returns:
            None

        """

        # Initial acceleration
        subdomains = self.mdg.subdomains(dim=self.nd)
        inertia_mass = self.inertia(subdomains)
        div = pp.ad.Divergence(subdomains, dim=self.nd)
        stress = pp.ad.Scalar(-1) * self.stress(subdomains)
        M_inv = 1 / inertia_mass.value(self.equation_system)
        Ku = (div @ stress).value(self.equation_system)
        initial_acceleration = M_inv * Ku

        data = self.mdg.subdomain_data(subdomains[0])

        pp.set_solution_values(
            name=self.acceleration_key,
            values=initial_acceleration,
            data=data,
            time_step_index=0,
            iterate_index=0,
        )
        
class DynamicMomentumBalanceLinearSpringModel(
    LinearSpringModel, DynamicMomentumBalanceABC
):
    """Dynamic momentum balance with linear spring-type fracture deformation."""


class DynamicMomentumBalanceBartonBandisSpringModel(
    BartonBandisSpringModel, DynamicMomentumBalanceABC
):
    """Dynamic momentum balance with Barton-Bandis spring-type fracture deformation."""
