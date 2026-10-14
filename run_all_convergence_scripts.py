import subprocess
import sys
from pathlib import Path

folder = Path(__file__).resolve().parent

scripts = [
    # Linear spring
    "runscript_convergence_linear_spring_space.py",
    "runscript_convergence_linear_spring_time.py",

    # Barton-Bandis
    "runscript_convergence_barton_bandis_spring_space.py",
    "runscript_convergence_barton_bandis_spring_time.py",

    # Frictional contact mechanics
    "runscript_convergence_frictional_contact_mechanics_space.py",
    "runscript_convergence_frictional_contact_mechanics_time.py",

    # Plotting
    "plot_self_convergence.py",
    "plot_self_convergence_all_refinements.py",
]

for script in scripts:
    script_path = folder / script

    print(f"\n{'=' * 70}", flush=True)
    print(f"Running: {script}", flush=True)
    print(f"{'=' * 70}", flush=True)

    if not script_path.is_file():
        raise FileNotFoundError(
            f"Script not found: {script_path}"
        )

    subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
        cwd=folder,
    )

print("\nAll convergence scripts completed successfully")
