"""Transfer Grid construction for P1 potential prolongation between grids.

A Transfer Grid is an intermediate grid built from the geometric intersection
of a source grid and a target grid.  It serves as the domain over which P1
potentials defined on the source are prolongated and then projected onto the
target via the Scott-Zhang quasi-interpolant (see ``primal_projections``).

This is needed in the non-matching estimator when the pressure trace from the
high-dimensional subdomain must be evaluated on a grid with different
connectivity than the mortar or fracture grid.

References
----------
Varela, J., Schaerer, C. E., Keilegavlen, E., & Berre, I. (2025).
A posteriori error estimates for mixed-dimensional Darcy flow using
non-matching grids. arXiv preprint.
https://doi.org/10.48550/arXiv.2512.09087
"""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
import scipy.sparse as sps
from matplotlib.collections import PolyCollection
from porepy.grids.refinement import structured_refinement
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from shapely.strtree import STRtree

import mdamr
from mdamr.utils.grid_utils import (
    ear_clip_triangulate,
    ensure_ccw,
    merge_close_vertices,
)


class TransferGrid:
    """
    Transfer grid between source and target grids.

    Assumes source and target lie on the same geometric plane. All geometric
    intersection and connectivity computations are done in their shared rotated
    2D parameterization (via mdamr.RotatedGrid), which is cached internally.

    """

    def __init__(
        self,
        g_source: pp.GridLike,
        g_target: pp.GridLike,
        rotation_matrix: np.ndarray | None = None,
        tol: float = 1e-9,
        name: str = "transfer",
    ):

        self.tol = tol
        """Geometric tolerance. Default is 1e-8."""

        self.name = name
        """Name of the transfer grid. Default is ``transfer''."""

        self.g_source = g_source
        """Source grid."""

        self.g_target = g_target
        """Target grid."""

        self.rot_matrix = rotation_matrix
        """Rotation matrix used to rotate `g_source` and `g_target`.

        If not given, the rotation matrix used will be the rotation matrix of
        `g_source`.
        """

        # Holders for rotated grids
        self._rot_matrix = rotation_matrix
        self._src_rot = None
        self._tgt_rot = None

        self._build_intersection_polygons()
        self._triangulate_intersections()
        self._assemble_transfer_grid()
        self._build_connectivity_matrices()

    def _get_rotated_grid(self, grid: pp.Grid):
        if grid is self.g_source:
            if self._src_rot is None:
                self._src_rot = (
                    mdamr.RotatedGrid(grid, self._rot_matrix)  # type:ignore
                    if self._rot_matrix is not None
                    else mdamr.RotatedGrid(grid)
                )
                # If not provided, adopt the source’s matrix so target uses the same
                if self._rot_matrix is None:
                    self._rot_matrix = self._src_rot.rotation_matrix  # type:ignore
            return self._src_rot
        if grid is self.g_target:
            if self._tgt_rot is None:
                if self._rot_matrix is None:
                    raise ValueError(
                        "TransferGrid needs a rotation_matrix or"
                        " a rotated source first."
                    )
                self._tgt_rot = mdamr.RotatedGrid(grid, self._rot_matrix)  # type:ignore
            return self._tgt_rot
        return mdamr.RotatedGrid(grid)

    def _extract_triangles(self, grid: pp.GridLike):
        """Extract triangles of source and target grids.

        Note:
        -----
            Geometric computations are done using rotated grids. Note that this
            assumes that both source and target grids represent the same surface in 3D
            space. To avoid expensive computations, no check is done to assure that this
            is indeed the case.

        """
        assert isinstance(grid, pp.Grid)
        grid_rot = self._get_rotated_grid(grid)  # rotate grid
        nodes = grid_rot.nodes  # retrieve nodes (from rotated grid)
        cn = grid.cell_nodes().tocsc()  # cell-nodes connectivity
        cn_arr = cn.indices.reshape((3, grid.num_cells), order="F")  # make it an array
        tris = []  # prepare list to retrieve triangles
        for i in range(grid.num_cells):
            idx = cn_arr[:, i]
            coords_xy = nodes[:2, idx].T
            tris.append((i, Polygon(coords_xy)))
        return tris

    def _build_intersection_polygons(self):
        """Build the intersection polygons between source and target grids.
        
        This method is modified from the original with help of ChatGPT. The original
        method was a tad too slow for very fine meshes.
        
        """
        source = [
            p for _, p in self._extract_triangles(self.g_source)
            if p.is_valid and p.area > self.tol
        ]

        target = [
            p for _, p in self._extract_triangles(self.g_target)
            if p.is_valid and p.area > self.tol
        ]

        prepared = [prep(p) for p in target]

        tree = STRtree(target)

        intersections = []

        for poly_s in source:

            candidate_idx = tree.query(poly_s)

            for idx in candidate_idx:

                poly_t = target[idx]
                prep_t = prepared[idx]

                if not prep_t.intersects(poly_s):
                    continue

                inter = poly_s.intersection(poly_t)

                if inter.area > self.tol:
                    intersections.append(inter)

        self._intersection_polys = intersections
        
    def _triangulate_intersections(self):
        all_triangles = []
        for poly in self._intersection_polys:
            raw = list(poly.exterior.coords)[:-1]
            coords = []
            for p in raw:
                if not coords or coords[-1] != p:
                    coords.append(p)
            if len(coords) == 3 and poly.area > self.tol:
                all_triangles.append(coords)
            else:
                tris = ear_clip_triangulate(coords, tol=self.tol)
                all_triangles.extend(tris)
        self._all_triangles = all_triangles

    def _assemble_transfer_grid(self):
        # Sanity check
        if not self._all_triangles:
            total_area = sum(p.area for p in self._intersection_polys)
            raise RuntimeError(
                f"No intersection triangles"
                f" (total intersection area={total_area:.3e}). "
                "Likely the source and target are not coplanar"
                " or overlap is degenerate."
            )
        raw_verts = []
        for tri in self._all_triangles:
            for p in tri:
                raw_verts.append((float(p[0]), float(p[1])))
        # initial cells (with duplicates)
        pt_to_idx = {}
        verts = []
        for tri in self._all_triangles:
            for p in tri:
                if p not in pt_to_idx:
                    pt_to_idx[p] = len(verts)
                    verts.append((float(p[0]), float(p[1])))
        cells = [[pt_to_idx[p] for p in tri] for tri in self._all_triangles]

        # merge nearly duplicate vertices
        coords_arr, cells_merged = merge_close_vertices(verts, cells, tol=self.tol)
        # enforce orientation
        cells_ccw = ensure_ccw(cells_merged, coords_arr)

        # Drop degenerate cells: repeated node indices (collapsed after merge) or
        # near-zero area (cross product not strictly positive).  These arise when
        # intersection slivers pass the shapely area filter but collapse under the
        # vertex-merge tolerance; pp.TriangleGrid rejects inconsistently oriented
        # cells so we must remove them before construction.
        area_tol = self.tol ** 2
        valid: list[list[int]] = []
        for tri in cells_ccw:
            i0, i1, i2 = tri
            if i0 == i1 or i1 == i2 or i0 == i2:
                continue
            p0, p1, p2 = coords_arr[:, i0], coords_arr[:, i1], coords_arr[:, i2]
            cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
            if cross > area_tol:
                valid.append(tri)
        if not valid:
            raise RuntimeError(
                "TransferGrid: all intersection triangles are degenerate after "
                f"vertex merge (tol={self.tol:.2e}).  Consider reducing tol."
            )

        # Deduplicate by vertex set: two slivers from different source-target
        # pairs can collapse to the same three vertex indices after merging.
        # Keeping both causes pp.TriangleGrid to see the same face twice with
        # the same orientation, failing the consistency check.
        seen: set[frozenset[int]] = set()
        deduped: list[list[int]] = []
        for tri in valid:
            key = frozenset(tri)
            if key not in seen:
                seen.add(key)
                deduped.append(tri)
        valid = deduped

        cells_arr = np.array(valid).T  # (3, Ncells)

        self.transfer = pp.TriangleGrid(coords_arr, cells_arr, name=self.name)
        self.transfer.compute_geometry()

    # ---- connectivity queries ----
    def _build_connectivity_matrices(self):
        """
        Builds and caches the four connectivity matrices:
            source_to_transfer,
            transfer_to_source,
            transfer_to_target,
            target_to_transfer

        All are binary (0/1) based on centroid-in-polygon containment/touching.
        """
        # --- prepare polygons and spatial indices ---
        # Source polygons
        src_tris = self._extract_triangles(self.g_source)
        src_polys = [poly for _, poly in src_tris]
        prepared_src = [prep(p) for p in src_polys]
        tree_src = STRtree(src_polys)
        # Map geometry -> index (works if query returns geometry)
        src_poly_to_idx = {id(p): i for i, p in enumerate(src_polys)}

        # Target polygons
        tgt_tris = self._extract_triangles(self.g_target)
        tgt_polys = [poly for _, poly in tgt_tris]
        prepared_tgt = [prep(p) for p in tgt_polys]
        tree_tgt = STRtree(tgt_polys)
        tgt_poly_to_idx = {id(p): i for i, p in enumerate(tgt_polys)}

        # --- allocate matrices ---
        n_src = self.g_source.num_cells
        n_tr = self.transfer.num_cells
        n_tgt = self.g_target.num_cells

        s2t = lil_matrix((n_src, n_tr), dtype=int)
        t2tgt = lil_matrix((n_tr, n_tgt), dtype=int)

        # --- source -> transfer ---
        # Use strict containment first; fall back to nearest-centroid for boundary
        # centroids that touch multiple cells (avoids multi-parent ambiguity).
        tr_centroids = self.transfer.cell_centers  # shape (>=2, n_tr)
        src_centroids = self.g_source.cell_centers  # shape (>=2, n_src)
        for j in range(n_tr):
            pt = Point(tr_centroids[0, j], tr_centroids[1, j])
            candidates = []
            for hit in tree_src.query(pt):
                if isinstance(hit, (int, np.integer)):
                    i = int(hit)
                else:
                    i = src_poly_to_idx.get(id(hit), None)
                    if i is None:
                        try:
                            i = src_polys.index(hit)
                        except ValueError:
                            continue
                if prepared_src[i].contains(pt):
                    candidates.append(i)
                elif prepared_src[i].touches(pt):
                    candidates.append(i)
            if len(candidates) == 1:
                s2t[candidates[0], j] = 1
            elif len(candidates) > 1:
                # Centroid is on a shared edge/vertex: pick nearest source centroid
                dists = [
                    (src_centroids[0, i] - tr_centroids[0, j]) ** 2
                    + (src_centroids[1, i] - tr_centroids[1, j]) ** 2
                    for i in candidates
                ]
                s2t[candidates[int(np.argmin(dists))], j] = 1

        # --- transfer -> source is transpose ---
        t2s = s2t.transpose().tocsr()

        # --- transfer -> target ---
        tgt_centroids = self.g_target.cell_centers
        for i in range(n_tr):
            pt = Point(tr_centroids[0, i], tr_centroids[1, i])
            candidates = []
            for hit in tree_tgt.query(pt):
                if isinstance(hit, (int, np.integer)):
                    j = int(hit)
                else:
                    j = tgt_poly_to_idx.get(id(hit), None)
                    if j is None:
                        try:
                            j = tgt_polys.index(hit)
                        except ValueError:
                            continue
                if prepared_tgt[j].contains(pt):
                    candidates.append(j)
                elif prepared_tgt[j].touches(pt):
                    candidates.append(j)
            if len(candidates) == 1:
                t2tgt[i, candidates[0]] = 1
            elif len(candidates) > 1:
                dists = [
                    (tgt_centroids[0, j] - tr_centroids[0, i]) ** 2
                    + (tgt_centroids[1, j] - tr_centroids[1, i]) ** 2
                    for j in candidates
                ]
                t2tgt[i, candidates[int(np.argmin(dists))]] = 1

        # --- target -> transfer is transpose ---
        tgt2t = t2tgt.transpose().tocsr()

        # Cache
        self.source_to_transfer = s2t.tocsr()
        self.transfer_to_source = t2s
        self.transfer_to_target = t2tgt.tocsr()
        self.target_to_transfer = tgt2t

    # NOTE: This is an experimental method, use it with caution
    @classmethod
    def from_nested(
        cls,
        g_source: pp.Grid,
        g_target: pp.Grid,
        coarse_fine: sps.csc_matrix | None = None,  # shape (n_fine, n_coarse)
        rotation_matrix: np.ndarray | None = None,  # kept for API symmetry; unused
        tol: float = 1e-8,
        name: str = "transfer",
    ) -> "TransferGrid":
        """
        Fast path for (assumed) nested refinement: use the *fine* grid as the transfer
        mesh, and assemble the 0/1 incidence matrices algebraically from a
        (fine × coarse) mapping.

        Equal-cell case:
          - Supported only if an explicit (n×n) mapping is provided.
          - Mapping may be identity or a permutation-like 0/1 matrix (one 1 per row).
          - In this case we treat g_source as "fine" and g_target as "coarse" by convention.
        """
        n_src, n_tgt = g_source.num_cells, g_target.num_cells

        def _is_valid_row_stochastic(M: sps.spmatrix) -> bool:
            # one 1 per row, 0/1 entries; columns can be >=1 for nested; for equal cells,
            # permutation would also have one 1 per column.
            row_sums = np.asarray(M.sum(axis=1)).ravel()
            return np.allclose(row_sums, 1.0)  # tolerate float format

        # ---- decide fine/coarse role and pick mapping M (fine × coarse) ----
        if n_src == n_tgt:
            # Equal-cell special: require an explicit mapping
            if coarse_fine is None:
                raise ValueError(
                    "from_nested: source and target have the same number of cells. "
                    "Please provide an explicit (n×n) coarse_fine mapping"
                    " (e.g., identity or permutation). Otherwise, build a geometric"
                    " TransferGrid instead."
                )
            M = coarse_fine.tocsc()
            if M.shape != (n_src, n_tgt):
                raise ValueError(
                    f"from_nested: provided mapping has shape {M.shape},"
                    f" expected {(n_src, n_tgt)}."
                )
            if not _is_valid_row_stochastic(M):
                raise ValueError(
                    "from_nested: mapping for equal-cell case must have exactly one"
                    " 1 per row."
                )
            # Convention: treat source as fine, target as coarse
            g_fine, g_coarse = g_source, g_target
            src_is_coarse = False
        else:
            # Strict nested by cell counts
            if n_src < n_tgt:
                g_coarse, g_fine = g_source, g_target
                src_is_coarse = True
            else:
                g_coarse, g_fine = g_target, g_source
                src_is_coarse = False

            # pick/find mapping if absent
            if coarse_fine is None:
                # try typical storage on coarse grid
                d = getattr(g_coarse, "data", {})
                M0 = d.get("coarse_fine_cell_mapping", None)
                if isinstance(M0, sps.spmatrix) and M0.shape == (
                    g_fine.num_cells,
                    g_coarse.num_cells,
                ):
                    M = M0.tocsc()
                else:
                    raise ValueError(
                        "from_nested: coarse_fine (fine×coarse) not provided and"
                        " not found in g_coarse.data['coarse_fine_cell_mapping']."
                    )
            else:
                M = coarse_fine.tocsc()
                if M.shape != (g_fine.num_cells, g_coarse.num_cells):
                    raise ValueError(
                        f"from_nested: provided mapping has shape {M.shape}, expected "
                        f"{(g_fine.num_cells, g_coarse.num_cells)}."
                    )

        # ---- construct a lightweight instance ----
        obj = cls.__new__(cls)
        obj.tol = tol
        obj.name = name
        obj.g_source = g_source
        obj.g_target = g_target

        # use the *actual* fine mesh as transfer
        obj.transfer = g_fine.copy()
        R_eff = mdamr.RotatedGrid(g_source).rotation_matrix
        obj._rot_matrix = R_eff
        obj._src_rot = None
        obj._tgt_rot = None

        # ---- assemble the four incidence matrices ----
        n_fine = g_fine.num_cells
        I_fine = sps.identity(n_fine, format="csc")

        if src_is_coarse:
            # source == coarse, target == fine
            s2t = M.T  # (n_coarse × n_fine)
            t2s = M  # (n_fine  × n_coarse)
            t2tgt = I_fine  # (n_fine  × n_fine)
            tgt2t = I_fine
        else:
            # source == fine, target == coarse
            s2t = I_fine
            t2s = I_fine
            t2tgt = M  # (n_fine × n_coarse)
            tgt2t = M.T

        obj.source_to_transfer = s2t.tocsr()
        obj.transfer_to_source = t2s.tocsr()
        obj.transfer_to_target = t2tgt.tocsr()
        obj.target_to_transfer = tgt2t.tocsr()

        return obj

    def summary(self):
        return {
            "n_source_cells": self.g_source.num_cells,
            "n_target_cells": self.g_target.num_cells,
            "n_transfer_cells": self.transfer.num_cells,
            "n_transfer_nodes": self.transfer.num_nodes,
        }

    def plot(self, ax=None, base_cmap="rainbow", alpha=1.0):
        """
        Plot the transfer mesh with a proper 4-coloring (no 2 neighbors share a color).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If None, a new figure+axes is created.
        base_cmap : str or Colormap, optional
            A Matplotlib colormap to draw from (should have >=4 distinct colors).
        alpha : float, optional
            Face alpha for the polygons.
        Returns
        -------
        fig, ax : tuple
            The figure and axes containing the plot.
        """
        # 1) prepare axes
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        # 2) get the 2D nodes and cells
        nodes2d = self.transfer.nodes[:2, :]
        cn = self.transfer.cell_nodes().tocsc()
        cells = cn.indices.reshape(
            (3, self.transfer.num_cells), order="F"
        ).T  # (n_tri, 3)

        # 3) build edge→triangles lookup
        edge_to_tris: dict[tuple[int, int], list[int]] = {}
        for t_idx, tri in enumerate(cells):
            for edge in combinations(tri, 2):
                e = tuple(sorted(edge))
                edge_to_tris.setdefault(e, []).append(t_idx)

        # 4) build adjacency list
        n_tri = len(cells)
        neighbors = [set() for _ in range(n_tri)]
        for tris in edge_to_tris.values():
            if len(tris) == 2:
                i, j = tris
                neighbors[i].add(j)
                neighbors[j].add(i)

        # 5) greedy graph-coloring
        colors = [-1] * n_tri
        for t in range(n_tri):
            used = {colors[nbr] for nbr in neighbors[t] if colors[nbr] >= 0}
            # assign smallest non-negative integer not in used
            c = 0
            while c in used:
                c += 1
            colors[t] = c
        n_colors = max(colors) + 1

        # 6) sample RGBA’s from colormap
        cmap = plt.get_cmap(base_cmap)
        # for categorical colors, take indices 0, 1/(n_colors-1),...,1
        color_vals = cmap(np.linspace(0, 1, n_colors))

        # 7) build the polygons
        verts = [nodes2d[:, tri].T for tri in cells]

        # 8) build collection with facecolors by triangle-color
        facecolors = [color_vals[c] for c in colors]
        coll = PolyCollection(
            verts,
            facecolors=facecolors,
            edgecolors="none",
            alpha=alpha,
        )
        ax.add_collection(coll)

        # 9) finalize
        ax.autoscale()
        ax.set_aspect("equal", "box")
        ax.set_xticks([])
        ax.set_yticks([])

        # 10) save figure
        fig = ax.get_figure()
        fig.savefig(f"{self.name}.pdf")


# ---- Utility functions ---
def build_transfer_grid_nested(
    gA: pp.Grid, gB: pp.Grid, mapping: sps.csc_matrix | None = None
) -> TransferGrid:
    """Return a TransferGrid using the fast nested path."""
    return TransferGrid.from_nested(gA, gB, coarse_fine=mapping, name="transfer_fast")


def coarse_fine_or_build(
    gA: pp.Grid, gB: pp.Grid, *, tol: float = 1e-9
) -> sps.csc_matrix:
    """
    Return coarse_fine mapping with shape (n_fine x n_coarse).

    - If g_coarse.data['coarse_fine_cell_mapping'] exists and matches shape, use it.
    - If n_fine == n_coarse, return identity (no call to structured_refinement).
    - Otherwise build via structured_refinement(g_coarse, g_fine).
    """
    # decide who is coarse/fine by num_cells
    if gA.num_cells <= gB.num_cells:
        g_coarse, g_fine = gA, gB
    else:
        g_coarse, g_fine = gB, gA

    n_coarse, n_fine = g_coarse.num_cells, g_fine.num_cells

    # 1) use cached if present and correct shape
    d = getattr(g_coarse, "data", None)
    if isinstance(d, dict) and "coarse_fine_cell_mapping" in d:
        M0 = d["coarse_fine_cell_mapping"]
        if isinstance(M0, sps.spmatrix) and M0.shape == (n_fine, n_coarse):
            return M0.tocsc()

    # 2) equal-size case: identity (fine x coarse) == (n x n)
    if n_fine == n_coarse:
        return sps.identity(n_fine, format="csc")

    # 3) strictly nested: build
    M = structured_refinement(g_coarse, g_fine, point_in_poly_tol=tol).tocsc()
    return M


def permute_transfer_columns(A: sps.spmatrix, perm: np.ndarray) -> sps.spmatrix:
    """Return A with its columns permuted so that new[:, j] = A[:, perm[j]]."""
    P = sps.coo_matrix(
        (np.ones(len(perm)), (perm, np.arange(len(perm)))), shape=(len(perm), len(perm))
    ).tocsr()
    return A @ P


def transfer_permutation_by_centroids(
    tg_ref, tg_to_perm, *, rtol=0, atol=1e-12
) -> np.ndarray:
    """
    Compute permutation that reorders tg_to_perm.transfer cells to match tg_ref.transfer
    by nearest neighbor matching of 2D centroids.
    """
    C_ref = tg_ref.transfer.cell_centers[:2, :].T
    C_perm = tg_to_perm.transfer.cell_centers[:2, :].T
    tree = cKDTree(C_perm)
    d, idx = tree.query(C_ref, k=1)
    if not np.all(d <= atol + rtol * np.abs(C_ref).max()):
        raise AssertionError(f"Transfer centroids mismatch; max diff {d.max():.3e}")
    return idx
