import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker

FONT_SIZE_1 = 18
FONT_SIZE_2 = 22

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": FONT_SIZE_2,
        "axes.titlesize": FONT_SIZE_2,
        "xtick.labelsize": FONT_SIZE_1,
        "ytick.labelsize": FONT_SIZE_1,
    }
)

FORCE_FLIP_PAIRS = {(1, 4), (3, 6)}

# Fracture geometry, given as endpoints in the format [[x0, x1], [y0, y1]].
# The lengths are computed from those endpoints after unit conversion.
FRACTURE_ENDPOINTS = {
    "fracture_0": np.array([[1, 4], [5.5, 7]]) * 25.0e-3 / 8,
    "fracture_1": np.array([[2, 3], [6.5, 5.0]]) * 25.0e-3 / 8,
    "fracture_2": np.array([[6, 7], [7, 6]]) * 25.0e-3 / 8,
    "fracture_3": np.array([[1, 4], [2.5, 1]]) * 25.0e-3 / 8,
    "fracture_4": np.array([[2, 3], [1.5, 3]]) * 25.0e-3 / 8,
    "fracture_5": np.array([[6, 7], [1, 2]]) * 25.0e-3 / 8,
}


def load_data(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    arrays = []
    for block in content.split("],"):
        block = block.strip()
        if not block:
            continue
        block = (
            block.replace("[", "").replace("]", "").replace(",", " ").replace("\n", " ")
        )
        arrays.append(np.fromstring(block, sep=" "))
    return np.vstack(arrays).astype(float)


def load_fracture(fracture_name, results_dir):
    d = os.path.join(results_dir, fracture_name)
    return {
        "normal": load_data(os.path.join(d, "displacement_jump_n.txt")),
        "slip": load_data(os.path.join(d, "slip_tendency.txt")),
    }


def build_axes(shape, fracture_name):
    nt, nx = shape

    t = np.linspace(0, 1.3e-5, nt)

    endpoints = FRACTURE_ENDPOINTS[fracture_name]
    x0, x1 = endpoints[0]
    y0, y1 = endpoints[1]

    length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    x = np.linspace(0, length, nx)

    return t, x


def maybe_flip(mat, pair):
    return mat[:, ::-1] if pair in FORCE_FLIP_PAIRS else mat


def plot_pair(frac_a, frac_b, results_dir):

    a = frac_a
    b = frac_b

    name_a = f"fracture_{a - 1}"
    name_b = f"fracture_{b - 1}"

    da = load_fracture(name_a, results_dir)
    db = load_fracture(name_b, results_dir)

    cmap = mpl.colormaps["viridis"].copy()
    cmap_pw = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad("white")

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 8), sharey=True, constrained_layout=True
    )

    fields = [("normal", r"$[\![u]\!]_n$"), ("slip", r"$s$")]

    for row, (key, label) in enumerate(fields):
        A = maybe_flip(da[key], (a, b))
        B = maybe_flip(db[key], (a, b))

        vmin = min(np.nanmin(A), np.nanmin(B))
        vmax = max(np.nanmax(A), np.nanmax(B))

        if key == "normal":
            m = max(
                np.nanmax(np.abs(A)),
                np.nanmax(np.abs(B))
            )

            vmin, vmax = -m, m

        if key == "normal":
            print(f"\nFracture pair {a}-{b} (normal displacement jump):")
            print(f"  Fracture {a}: min = {np.nanmin(A):.6e}, max = {np.nanmax(A):.6e}")
            print(f"  Fracture {b}: min = {np.nanmin(B):.6e}, max = {np.nanmax(B):.6e}")

        last_im = None

        for col, (num, mat) in enumerate([(a, A), (b, B)]):
            ax = axes[row, col]

            t, x = build_axes(mat.shape, name_a if col == 0 else name_b)

            im = ax.pcolormesh(
                x,
                t,
                mat,
                shading="auto",
                cmap=cmap_pw if key == "normal" else cmap,
                vmin=vmin,
                vmax=vmax,
            )

            last_im = im

            if row == 0:
                ax.set_title(f"Fracture {num}", fontsize=FONT_SIZE_2)

            if row == 1:
                ax.set_xlabel("Position along fracture [m]")

            if col == 0:
                ax.set_ylabel("Time [s]")

            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        if key == "slip":
            pad = 0.068
        else:
            pad = 0.01
        cbar = fig.colorbar(
            last_im, ax=axes[row, :], location="right", pad=pad, shrink=0.95
        )
        cbar.set_label(label, fontsize=FONT_SIZE_2)
        cbar.ax.tick_params(labelsize=FONT_SIZE_1)

    out = os.path.join(results_dir, "figures", f"fracture_pair_{a}_{b}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", out)


if __name__ == "__main__":
    results_dir = "simulation_example_results_2d"
    for pair in [(1, 4), (2, 5), (3, 6)]:
        plot_pair(*pair, results_dir)
