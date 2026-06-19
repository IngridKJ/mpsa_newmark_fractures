# mpsa_newmark_fractures# momentum_balance_inertia
This is the "working repository" for our work on the MPSA-Newmark method, meaning that te content of this repository is very much alive and evolving.

This readme describes most of the contents in the repository, and is structured as follows:
* Models

* Verification (fractureless)

  * Convergence analysis: Homogeneous boundary conditions 
  * Convergence analysis: Absorbing boundary conditions
  * Energy decay analysis

* Verification (with fractures)
  * Linear spring-type deformation
  * Barton-Bandis spring-type deformation
  * Fracture contact mechanics with friction

* Simulation examples
  * From paper 1: Anisotropic and heterogeneous media
  * From the newest work: Multiple intersecting fractures

* Utility material
* Tests

## Models
The model classes are standardized setups for solving the elastic wave equation with absorbing boundary conditions with PorePy. All model class setups, apart from those tailored to specific setups, are found within [elastic_wave_equation_abc.py](./models/elastic_wave_equation_abc.py) and follow the naming-convention `DynamicMomentumBalance` with some descriptive suffix. The module contains the following setups:
* A general setup for solving the elastic wave equation with absorbing boundaries on all domain sides in a fractureless domain
* Different setups for fractured media:
  * Fracture contact mechanics with friction
  * Fracture contact mechanics with friction, but parts of the tangential fracture equation is smoothed out
  * Two spring-type deformation models

## Verification (fractureless)
All runscripts and model setups referred to in tihs section are from our [first publication](https://doi.org/10.5149/ARC-GR.1598). All scripts are modified a bit since publication due to updates in both PorePy and this repository, and I therefore recommend to check out the [Zenodo upload](https://doi.org/10.5281/zenodo.15056461) for a devcontainer of the full setup. The repository from the paper is found [here](https://github.com/IngridKJ/mpsa_newmark).
### Convergence analysis: Homogeneous boundary conditions
The convergence analyses presented in the article are performed with 
homogeneous Dirichlet conditions on a 3D simplex grid:
* Convergence in space and time:
  * [runscript_space_time_convergence_dirichlet_boundaries](./convergence_and_stability_analysis_paper_1/runscript_space_time_convergence_dirichlet_boundaries.py)
* Convergence in space:
  * [runscript_space_convergence_dirichlet_boundaries](./convergence_and_stability_analysis_paper_1/runscript_space_convergence_dirichlet_boundaries.py) 
* Convergence in time:
  * [runscript_time_convergence_dirichlet_boundaries](./convergence_and_stability_analysis_paper_1/runscript_time_convergence_dirichlet_boundaries.py) 

All the runscripts utilize
[manufactured_solution_dynamic_3D](./convergence_and_stability_analysis_paper_1/analysis_models/manufactured_solution_dynamic_3D.py)
as the manufactured solution setup.

### Convergence analysis: absorbing boundaries
Convergence of the solution is performed by sending an orthogonal wave from the left
towards the right boundary, where the right boundary is absorbing. The top and bottom
boundaries have roller conditions, and the left boundary is a time-dependent Dirichlet
condition which maintains the orthogonal wave after initialization.

Convergence in space and time, with the media being isotropic/anisotropic and homogeneous/heterogeneous:
  * [runscript_space_time_convergence_absorbing_boundaries](./convergence_and_stability_analysis_paper_1/runscript_space_time_convergence_absorbing_boundaries.py)
    which uses the model class setup found in
    [model_convergence_ABC](./convergence_and_stability_analysis_paper_1/analysis_models/model_convergence_ABC.py)


### Energy decay analysis
The energy decay analysis is performed both for successive refinement 
of the grid, as well as for varying wave incidence angles. 

* Successive grid refinement is done by running the script
[runscript_energy_decay_space_refinement](./convergence_and_stability_analysis_paper_1/runscript_energy_decay_space_refinement.py).

* Varying the wave incidence angle, $\theta$, is done by running the script
  [runscript_energy_decay_vary_theta](./convergence_and_stability_analysis_paper_1/runscript_energy_decay_vary_theta.py).

## Verification (with fractures)  
The files mentioned in this section are from the work on inclusion of fractures. We consider different types of fracture deformation, where all models are verified through numerical convergence analyses.
### Linear spring-type deformation
Convergence against a known analytical solution for a fractured medium, similar to the convergence setup with absorbing boundaries mentioned above. In this case there is a fracture located in the middle of the domain and at the intersection between the two (potentially) dissimilar media.
* Runscript: 
  * [runscript_convergence_linear_spring](./runscript_convergence_linear_spring.py)
* The model class setup:
  * [model_convergence_linear_spring](./model_convergence_linear_spring.py)

### Barton-Bandis spring type deformation
Convergence of the transmission coefficient. We perform a comparison between the theoretical and numerical peak velocity after a wavetop has crossed a fracture.
* Runscript: 
  * [runscript_convergence_barton_bandis_spring](./runscript_convergence_barton_bandis_spring.py)
* The model class setup is named `SpringTypeBartonBandisConvergenceSetup` and is found within:
  * [models_nonlinear_fracture_deformation](./models_nonlinear_fracture_deformation.py)

### Fracture contact mechanics with friction (WIP)
The fracture contact mechanics with friction is verified by self-convergence analyses for both the linear and the Barton-Bandis gap function. The wave is imposed by a time-dependent Dirichlet condition on the left boundary and eventually hits a diagonal fracture.
* Runscript:
  * [runscript_convergence_contact_mechanics](./runscript_convergence_contact_mechanics.py)
* The model class setups are named `ContactModelBartonBandisGapFunction` and `ContactModelLinearGapFunction`, and are found within:
  * [models_nonlinear_fracture_deformation](./models_nonlinear_fracture_deformation.py)

## Simulation examples
### From paper 1: Anisotropic and heterogeneous media
Simulation example runscripts for paper 1 are found within [this](./example_runscripts_paper_1/) folder.
* The simulation from Example 1.1, which considers a seismic source located inside an
  inner transversely isotropic domain, is run by
  [runscript_example_1_1_source_in_inner_domain](./example_runscripts_paper_1/runscript_example_1_1_source_in_inner_domain.py).
* The simulation from Example 1.2, which considers a seismic source located outside an
  inner transversely isotropic domain, is run by
  [runscript_example_1_2_source_in_outer_domain](./example_runscripts_paper_1/runscript_example_1_2_source_in_outer_domain.py).
* The simulation from Example 2, which considers a layered heterogeneous medium with an
  open fracture, is run by
  [runscript_example_2_heterogeneous_fractured_domain](./example_runscripts_paper_1/runscript_example_2_heterogeneous_fractured_domain.py).
### From the newest work: Fracture contact mechanics with friction
The simulation example has one 2D version and one 3D version. The 3D setup is the 2D one but expanded in the z-direction. 

We consider a fractured media where the fractures are geometrically symmetric but hold different fracture parameters. The runscripts for the simulation example are:
* [runscript_example_symmetric_fractures](./runscript_example_symmetric_fractures.py)
* [runscript_example_symmetric_fractures_3d](./runscript_example_symmetric_fractures_3d.py)

where both of them utilize the `ContactModelBartonBandisGapFunction` setup found in [this](./models_nonlinear_fracture_deformation.py) file.

## Utility material
A collection of utility material is found within the [utils](./utils/) directory:
* [anisotropy mixins](./utils/anisotropy_mixins.py) contains mixins for anisotropic
stiffness tensors.
* [perturbed_geometry_mixins](./utils/perturbed_geometry_mixins.py) contains mixins for
three types/configurations of perturbed geometry. This is not really needed anymore after PorePy 1.13.
* [utility functions](./utils/utility_functions.py) contains mostly functions related to
analytical solution expressions and fetching subdomain-related quantities. Utility
functions for defining the stiffness tensor for a transversely isotropic media are also
to be found found here.

I refer to the files within the directory for more details about the specific contents.

## Tests
Tests are covering:
* MPSA-Newmark convergence:
  * With homogeneous Dirichlet conditions in 2D and 3D.
  * With absorbing boundaries in homogeneous and heterogeneous media (with and without fracture).
* Construction of the transversely isotropic tensor.
* The utility functions ``inner_domain_cells``
  ``use_constraints_for_inner_domain_cells`` which can be used in the construction of
  the transversely isotropic tensor.
* The generation of a heterogeneous stiffness tensor.