import porepy as pp
import numpy as np


class LinearSpringModel:
    def _is_nonlinear_problem(self) -> bool:
        """Asserting problem is nonlinear."""
        return True
    
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        bc = super().bc_type_mechanics(sd=sd)
        bc.internal_to_dirichlet(sd)
        return bc

    def tangential_fracture_deformation_equation(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """The tangential component of the fracture deformation equation.

        Fracture deformation is modelled by what is often referred to as the linear slip
        model (LSM). The fracture deformation is linear in both the tangential and
        normal direction. This method implements the tangential component of the linear
        spring-type fracture deformation model.

        Linear spring-type fracture deformation model:
        .. math::
            [u_t] = Z_t * t_t,
        where :math:`t_t` is the tangential component of the contact traction and
        :math:`Z_t` is the tangential fracture compliance. The symbol :math:`[u_t]`
        denotes the tangential displacement jump.

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            The operator the for tangential fracture deformation equation described by
            the linear deformation model.

        """
        nd_vec_to_tangential = self.tangential_component(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)

        equation: pp.ad.Operator = u_t - self.elastic_tangential_fracture_deformation(
            subdomains
        )
        equation.set_name("tangential_fracture_deformation_equation")
        return equation

    def normal_fracture_deformation_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """The normal component of the fracture deformation equation.

        Fracture deformation is modelled by what is often referred to as the linear slip
        model (LSM). The fracture deformation is linear in both the tangential and
        normal direction. This method implements the normal component of the linear
        spring-type fracture deformation model.

        Linear spring-type fracture deformation model:
        .. math::
            [u_n] = Z_n * t_n,
        where :math:`t_n` is the normal component of the contact traction and
        :math:`Z_n` is the normal fracture compliance. The symbol :math:`[u_n]`
        denotes the normal displacement jump.

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            The operator the for normal fracture deformation equation described by
            the linear deformation model.

        """
        nd_vec_to_normal = self.normal_component(subdomains)
        u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump(subdomains)

        equation: pp.ad.Operator = u_n - self.elastic_normal_fracture_deformation(
            subdomains
        )
        equation.set_name("normal_fracture_deformation_equation")
        return equation

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

class BartonBandisSpringModel(LinearSpringModel):
    """The methods for Barton Bandis is copy pasted in here. I had some MRO troubles."""

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
        # The maximum opening of the fracture.
        maximum_opening = self.maximum_elastic_fracture_opening(subdomains)

        # If the maximum opening is zero, the Barton-Bandis model is not valid in the
        # case of zero normal traction. In this case, we return an empty operator.
        # If the maximum opening is negative, an error is raised.
        val = self.equation_system.evaluate(maximum_opening)
        if np.any(val == 0):
            num_cells = sum(sd.num_cells for sd in subdomains)
            return pp.ad.DenseArray(np.zeros(num_cells), "zero_Barton-Bandis_opening")
        elif np.any(val < 0):
            raise ValueError("The maximum opening must be non-negative.")

        nd_vec_to_normal = self.normal_component(subdomains)

        # The scaled effective contact traction [-]. The papers by Barton and Bandis
        # assume positive traction in contact, thus we need to switch the sign.
        contact_traction = pp.ad.Scalar(-1) * self.contact_traction(subdomains)

        # Normal component of the traction.
        normal_traction = nd_vec_to_normal @ contact_traction

        # Normal stiffness (as per Barton-Bandis terminology). Units: Pa / m.
        normal_stiffness = self.fracture_normal_stiffness(subdomains)
        # Rescale, since contact traction is dimensionless.
        scaled_stiffness = normal_stiffness / self.characteristic_contact_traction(
            subdomains
        )

        # The opening is found from the 1983 paper.
        opening_decrease = (
            normal_traction
            * maximum_opening
            / (scaled_stiffness * maximum_opening + normal_traction)
        )
        elastic_opening = -opening_decrease
        elastic_opening.set_name("Barton-Bandis_elastic_opening")
        return elastic_opening

    def maximum_elastic_fracture_opening(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """The maximum opening of a fracture [m].

        Used in the Barton-Bandis model for normal elastic fracture deformation.

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            The maximum allowed increase in fracture opening.

        """
        max_opening = self.solid.maximum_elastic_fracture_opening
        return pp.ad.Scalar(max_opening, "maximum_elastic_fracture_opening")

    def fracture_normal_stiffness(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """The normal stiffness of a fracture [Pa*m^-1].

        Used in the Barton-Bandis model for normal elastic fracture deformation.

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            The fracture normal stiffness.

        """

        normal_stiffness = self.solid.fracture_normal_stiffness
        return pp.ad.Scalar(normal_stiffness, "fracture_normal_stiffness")
