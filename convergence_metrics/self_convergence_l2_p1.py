"""Utilities for L2 error computation in self-convergence studies on non-matching grids.

Assumptions:
* 2D computations assume conforming triangular meshes .
* 1D computations assume structured nested refinement of line segments.
* TransferGrid is assumed to produce a one-to-one mapping between transfer cells
  and parent coarse/fine cells.
* All reconstructions are P1 (affine per cell) and evaluated exactly using
  reference mass matrices (no numerical quadrature).

This module is intended for reproducing the self-convergence studies presented in the
accompanying paper distributed with this code. It is not designed as a general-purpose
numerical library, and relies on strict mesh and discretization assumptions consistent
with those experiments.

"""

from __future__ import annotations

import numpy as np
import porepy as pp

from mdamr.non_matching.transfer_grid import TransferGrid


def _cell_nodes(sd: pp.Grid, expected: int) -> np.ndarray:
    cn = sd.cell_nodes().tocsc()
    out = []
    for c in range(sd.num_cells):
        nodes = cn.indices[cn.indptr[c] : cn.indptr[c + 1]]
        if len(nodes) != expected:
            raise ValueError(f"Cell {c} has {len(nodes)} nodes, expected {expected}.")
        out.append(nodes)
    return np.asarray(out, dtype=int)


def _nodal_average_from_cell_values(
    sd: pp.Grid,
    values_cc: np.ndarray,
    weight: str = "volume",
) -> np.ndarray:
    """
    Patchwise P0 -> nodal reconstruction.

    values_cc:
        shape (n_components, num_cells) or shape (num_cells,) for scalar.

    """
    values_cc = np.asarray(values_cc)

    scalar = values_cc.ndim == 1
    if scalar:
        values_cc = values_cc[None, :]

    n_comp, n_cells = values_cc.shape
    if n_cells != sd.num_cells:
        raise ValueError(f"values_cc has {n_cells} cells, grid has {sd.num_cells}.")

    u_node = np.zeros((n_comp, sd.num_nodes))
    w_node = np.zeros(sd.num_nodes)

    expected = sd.dim + 1
    cells = _cell_nodes(sd, expected=expected)

    for c, nodes in enumerate(cells):
        if weight == "volume":
            w = float(sd.cell_volumes[c])
        elif weight == "uniform":
            w = 1.0
        else:
            raise ValueError("weight must be 'volume' or 'uniform'.")

        u_node[:, nodes] += w * values_cc[:, c, None]
        w_node[nodes] += w

    if np.any(w_node <= 0):
        raise RuntimeError("Some nodes received no cell contributions.")

    u_node /= w_node[None, :]

    return u_node[0] if scalar else u_node


def _p1_coefficients_from_nodal_values_2d(
    sd: pp.Grid,
    nodal_values: np.ndarray,
    node_xy: np.ndarray,
) -> np.ndarray:
    """Build cell-wise affine coefficients in 2D.

    Parameters:
        nodal_values: The values on all nodes in the grid. The array has shape
            (n_components, num_nodes)
        node_xy: Array of shape (2, num_nodes) with the x and y coordinates of the
            nodes. In the same coordinate frame used by TransferGrid.

    Returns
        coeff: Array with shape (n_components, num_cells, 3). The coefficients are coeff
        [comp, cell] = [a, b, c] and are used to represent u(x,y) = a*x + b*y + c.

    """
    nodal_values = np.asarray(nodal_values)
    if nodal_values.ndim == 1:
        nodal_values = nodal_values[None, :]

    n_comp = nodal_values.shape[0]
    cells = _cell_nodes(sd, expected=3)

    coeff = np.zeros((n_comp, sd.num_cells, 3))

    for c, nodes in enumerate(cells):
        xy = node_xy[:, nodes]  # (2, 3)
        A = np.column_stack((xy[0], xy[1], np.ones(3)))  # (3, 3)

        for m in range(n_comp):
            rhs = nodal_values[m, nodes]
            coeff[m, c, :] = np.linalg.solve(A, rhs)

    return coeff


def _eval_p1_2d(coeff: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Evaluate affine field at points.

    Parameters:
        coeff: Linear reconstruction coefficient array of shape (n_components, 3). 
            Coefficients for one cell.
        xy: Array of shape (2, n_points) with the x and y coordinates of the points.
            The xy array has shape (2, n_points)

    Returns
        values: The reconstructed field values at nodes xy. The array has shape
            (n_components, n_points).
    """
    return (
        coeff[:, 0:1] * xy[0][None, :] + coeff[:, 1:2] * xy[1][None, :] + coeff[:, 2:3]
    )


def _triangle_area(xy: np.ndarray) -> float:
    p0 = xy[:, 0]
    p1 = xy[:, 1]
    p2 = xy[:, 2]
    return 0.5 * abs(np.linalg.det(np.column_stack((p1 - p0, p2 - p0))))


def _transfer_triangles(tr: pp.Grid) -> np.ndarray:
    cn = tr.cell_nodes().tocsc()
    return cn.indices.reshape((3, tr.num_cells), order="F").T


def matrix_overlap_l2_components_p1(
    sd_H: pp.Grid,
    sd_h: pp.Grid,
    u_H: np.ndarray,
    u_h: np.ndarray,
    weight: str = "volume",
    tol: float = 1e-10,
) -> tuple[float, float]:
    """L2 overlap integral between two 2D vector P1 fields on a non-matching mesh.

    Parameters:
        sd_H: Coarse grid
        sd_h: Fine grid
        u_H: Coarse grid cell values
        u_h: Fine grid cell values
        weight: "volume" or "uniform" for nodal averaging
        tol: Tolerance for transfer grid construction

    Computes
        numerator_sq   = \\int_Omega |u_H^P1 - u_h^P1|^2 dx,
        denominator_sq = \\int_Omega |u_h^P1|^2 dx,
    on the transfer grid generated by intersections of coarse and fine cells.

    """
    if sd_H.dim != 2 or sd_h.dim != 2:
        raise ValueError("matrix_overlap_l2_components_p1 expects 2D grids.")

    u_H = np.asarray(u_H)
    u_h = np.asarray(u_h)

    if u_H.shape != (2, sd_H.num_cells):
        raise ValueError(f"uH_cc shape {u_H.shape} is not (2, {sd_H.num_cells}).")
    if u_h.shape != (2, sd_h.num_cells):
        raise ValueError(f"uh_cc shape {u_h.shape} is not (2, {sd_h.num_cells}).")

    transfer_grid = TransferGrid(sd_H, sd_h, tol=tol, name="matrix_transfer")

    # Important: use the same rotated coordinates as TransferGrid.
    source_grid_rotated = transfer_grid._get_rotated_grid(sd_H)
    target_grid_rotated = transfer_grid._get_rotated_grid(sd_h)

    uH_node = _nodal_average_from_cell_values(sd_H, u_H, weight=weight)
    uh_node = _nodal_average_from_cell_values(sd_h, u_h, weight=weight)

    CH = _p1_coefficients_from_nodal_values_2d(
        sd_H, uH_node, source_grid_rotated.nodes[:2, :]
    )
    Ch = _p1_coefficients_from_nodal_values_2d(
        sd_h, uh_node, target_grid_rotated.nodes[:2, :]
    )

    tr = transfer_grid.transfer
    tr_cells = _transfer_triangles(tr)
    Xtr = tr.nodes[:2, :]

    # Exact P1 mass matrix on a triangle:
    # \\int_T phi_i phi_j dx = |T|/12 * [[2,1,1],[1,2,1],[1,1,2]]
    M_ref = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) / 12.0

    numerator_sq = 0.0
    denominator_sq = 0.0

    transfer_to_source = transfer_grid.transfer_to_source.tocsr()
    transfer_to_target = transfer_grid.transfer_to_target.tocsr()

    for j, nodes in enumerate(tr_cells):
        xy = Xtr[:, nodes]
        area = _triangle_area(xy)

        source_parent = transfer_to_source[j].indices
        target_parent = transfer_to_target[j].indices

        if source_parent.size != 1 or target_parent.size != 1:
            raise RuntimeError(
                f"Transfer triangle {j} has source parents {source_parent} "
                f"and target parents {target_parent}; expected one each."
            )

        KH = int(source_parent[0])
        kh = int(target_parent[0])

        uH_v = _eval_p1_2d(CH[:, KH, :], xy)  # (2, 3)
        uh_v = _eval_p1_2d(Ch[:, kh, :], xy)  # (2, 3)

        diff_v = uH_v - uh_v

        M = area * M_ref

        for comp in range(2):
            numerator_sq += float(diff_v[comp] @ M @ diff_v[comp])
            denominator_sq += float(uh_v[comp] @ M @ uh_v[comp])

    return numerator_sq, denominator_sq


def _p1_coefficients_from_nodal_values_1d(
    sd: pp.Grid,
    nodal_values: np.ndarray,
    node_s: np.ndarray,
) -> np.ndarray:
    """Cell-wise affine coefficients on a 1D line.

    Parameters:
        nodal_values: shape (n_components, num_nodes) or (num_nodes,)

    Returns:
        coeff shape (n_components, num_cells, 2)
        coeff[m, c] = [a, b], u_m(s) = a*s + b.
    """
    nodal_values = np.asarray(nodal_values)

    if nodal_values.ndim == 1:
        nodal_values = nodal_values[None, :]

    n_comp = nodal_values.shape[0]
    cells = _cell_nodes(sd, expected=2)

    coeff = np.zeros((n_comp, sd.num_cells, 2))

    for c, (n0, n1) in enumerate(cells):
        s0 = float(node_s[n0])
        s1 = float(node_s[n1])

        if abs(s1 - s0) < 1e-14:
            raise RuntimeError(f"Degenerate 1D cell {c}.")
        for m in range(n_comp):
            v0 = float(nodal_values[m, n0])
            v1 = float(nodal_values[m, n1])

            a = (v1 - v0) / (s1 - s0)
            b = v0 - a * s0

            coeff[m, c, :] = [a, b]
    return coeff


def _eval_p1_1d(coeff: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    coeff:
        shape (n_components, 2)

    s:
        shape (n_points,)

    Returns:
        shape (n_components, n_points)
    """
    return coeff[:, 0:1] * s[None, :] + coeff[:, 1:2]


def fracture_overlap_l2_components_p1(
    sd_H: pp.Grid,
    sd_h: pp.Grid,
    wH_cc: np.ndarray,
    wh_cc: np.ndarray,
    weight: str = "volume",
) -> tuple[float, float]:
    """Simplified overlap integration for 1D fracture vector variable.

    Assumes sd_h is a nestedly refined version of sd_H.

    Parameters:
        sd_H: Coarse grid (1D)
        sd_h: Fine grid (1D)
        wH_cc: Coarse grid cell values
        wh_cc: Fine grid cell values
        weight: "volume" or "uniform" for nodal averaging

    Computes:
        numerator_sq   = \\int_\\Gamma |w_H^P1 - w_h^P1|^2 ds
        denominator_sq = \\int_\\Gamma |w_h^P1|^2 ds
    """
    if sd_H.dim != 1 or sd_h.dim != 1:
        raise ValueError("fracture_overlap_l2_components_p1 expects 1D grids.")

    wH_cc = np.asarray(wH_cc)
    wh_cc = np.asarray(wh_cc)

    if wH_cc.ndim == 1:
        wH_cc = wH_cc[None, :]
    if wh_cc.ndim == 1:
        wh_cc = wh_cc[None, :]

    if wH_cc.shape[1] != sd_H.num_cells:
        raise ValueError(
            f"wH_cc shape {wH_cc.shape} incompatible with {sd_H.num_cells} cells."
        )
    if wh_cc.shape[1] != sd_h.num_cells:
        raise ValueError(
            f"wh_cc shape {wh_cc.shape} incompatible with {sd_h.num_cells} cells."
        )

    # Reconstruct P1 fields at nodes
    wH_node = _nodal_average_from_cell_values(sd_H, wH_cc, weight=weight)
    wh_node = _nodal_average_from_cell_values(sd_h, wh_cc, weight=weight)

    # Get 1D parametrization (arc length coordinate)
    sH = sd_H.nodes[0, :]
    sh = sd_h.nodes[0, :]

    # Build P1 coefficients: u(s) = a*s + b
    CH = _p1_coefficients_from_nodal_values_1d(sd_H, wH_node, sH)
    Ch = _p1_coefficients_from_nodal_values_1d(sd_h, wh_node, sh)

    # Exact 1D P1 mass matrix on segment [s0, s1]:
    # \\int phi_i phi_j ds = L/6 * [[2,1],[1,2]]
    M_ref = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0

    numerator_sq = 0.0
    denominator_sq = 0.0

    # Iterate over fine cells and compute contributions
    for j in range(sd_h.num_cells):
        s0 = float(sh[j])
        s1 = float(sh[j + 1])
        length = abs(s1 - s0)

        # Find which coarse cell contains this fine cell
        # For nested refinement, the fine cell is inside exactly one coarse cell
        mid_point = 0.5 * (s0 + s1)

        # Find coarse cell containing mid_point
        coarse_cell_id = None
        for K in range(sd_H.num_cells):
            s_coarse_0 = float(sH[K])
            s_coarse_1 = float(sH[K + 1])
            if min(s_coarse_0, s_coarse_1) <= mid_point <= max(s_coarse_0, s_coarse_1):
                coarse_cell_id = K
                break

        if coarse_cell_id is None:
            raise RuntimeError(f"Fine cell {j} does not map to any coarse cell.")

        s_pair = np.array([s0, s1])

        # Evaluate P1 fields at endpoints
        wH_v = _eval_p1_1d(CH[:, coarse_cell_id, :], s_pair)  # (n_comp, 2)
        wh_v = _eval_p1_1d(Ch[:, j, :], s_pair)  # (n_comp, 2)

        diff_v = wH_v - wh_v
        M = length * M_ref

        for comp in range(diff_v.shape[0]):
            numerator_sq += float(diff_v[comp] @ M @ diff_v[comp])
            denominator_sq += float(wh_v[comp] @ M @ wh_v[comp])

    return numerator_sq, denominator_sq
