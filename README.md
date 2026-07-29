# mpsa_newmark_fractures
This repository contains runscripts and model class setups needed for reproducing the results in the paper "Modeling of elastic wave propagation in fractured media with spring-type and frictional contact deformation models".

That includes:
* Runscripts for the convergence analyses.
* Runscripts for all simulation examples.
* Standardized model class setup for solving the elastic wave equation in media with deforming fractures using PorePy (https://github.com/pmgbergen/porepy).
* Utility material which is used in the various simulations.

Some run scripts support one or more of the following optional parameters:
* `COARSE = True`: Run the model using a coarser grid and (possibly) larger time step. Setting `COARSE = False` uses the grid and time-step sizes reported in the paper.
* `SAVE_FIGURES = True`: Generate and save figures.
* `RUN_MODELS = False`: Skip model execution and generate plots from existing result files. NOTE: This option requires that the corresponding runscript has been executed previously.

Not all run scripts support all of these parameters.

## Verification 
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

### Fracture contact mechanics with friction
The fracture contact mechanics with friction is verified by self-convergence analyses for both the linear and the Barton-Bandis gap function. The wave is imposed by a time-dependent Dirichlet condition on the left boundary and eventually hits a diagonal fracture.
* Runscript:
  * [runscript_convergence_frictional_contact_mechanics](./runscript_convergence_frictional_contact_mechanics.py)
* The model class setups are named `ContactModelBartonBandisGapFunction` and
  `ContactModelLinearGapFunction`, and are found within:
  * [models_nonlinear_fracture_deformation](./models_nonlinear_fracture_deformation.py)

### Verification results
Upon running the different verification setups, figures, exported quantities and errors are saved in a directory with the name `convergence_analysis_results`.

## Simulation examples
We consider three simulation examples: one comparing four fracture deformation models, and two (2D and 3D) which considers a domain with multiple intersecting fractures.

In the model comparison example, the domain consists of one diagonal fracture governed by four different fracture deformation models:
* [runscript_example_model_comparison](./runscript_example_model_comparison.py)
where the model class setups are found in [this](./model_example_compare_models.py) file. Upon running the simulation example, figures and exported quantities are saved in a directory with the name `simulation_example_results_compare_models`.

In the example with multiple fractures, we consider a fractured media where the fractures are geometrically symmetric but hold different fracture parameters. The runscripts for the simulation example are:
* [runscript_example_symmetric_fractures_2d](./runscript_example_symmetric_fractures_2d.py)
* [runscript_example_symmetric_fractures_3d](./runscript_example_symmetric_fractures_3d.py)

where both of them utilize the `ContactModelBartonBandisGapFunction` setup found in [this](./models_nonlinear_fracture_deformation.py) file. Upon running the simulation examples, figures and exported quantities are saved in a directory with the name `simulation_example_results_2d` or `simulation_example_results_3d`, depending on the dimensionality of the problem.

Utility material:
* The [mdamr](./mdamr) directory contains utility material for comparing solutions on nonmatching grids (Varela et al. (2025). A posteriori error estimates for mixed-dimensional Darcy flow using non-matching grids. arXiv preprint. https://doi.org/10.48550/arXiv.2512.09087)
* The [utils](./utils) directory contains various utility functions such as constitutive laws, helper functions/classes for convergence analyses and more.