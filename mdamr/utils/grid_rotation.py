"""The content in this file is from the mdamr package by Jhabriel Varela."""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
import porepy as pp

import mdamr  # where RotatedGrid lives

GridLike = Union[pp.Grid, pp.MortarGrid]
Key = Tuple[str, int]  # ("sd", sd.id) or ("intf", intf.id)


# global caches (module-level)
_CANONICAL_FRAMES: Dict[Key, Tuple[np.ndarray | None, np.ndarray]] = {}
_ROTATED_CACHE: Dict[Key, mdamr.RotatedGrid] = {}


def _key(g: GridLike) -> Key:
    if isinstance(g, pp.MortarGrid):
        kind = "intf"
    else:
        kind = "sd"
    return (kind, g.id)


class RotatedGrid:
    """Parent class for rotated grid object."""

    def __init__(
        self,
        sd: pp.Grid,
        rotation_matrix: np.ndarray | None = None,
        tol=1e-10,  # same as `map_geometry.map_grid()` by default
    ):
        """
        This class is a wrapper of the outputs of pp.map_geometry.map_grid(sd).

        Alternatively, if `rotation_matrix` (3x3) is supplied, we use it to rotate
        the subdomain grid `sd`.

        Public Attributes:
        -------------------
            sd: Original pp.Grid
            cell_centers: Cell centers coordinates of the rotated grid.
            face_normals: Face normals of the rotated grid.
            face_centers: Face centers coordinates of the rotated grid.
            rotation_matrix: Rotation matrix used to map from the original to the
                rotated domain.
            dim_bool: Array of booleans, where True refers to an active dimension.
            dim: Effective dimension of the rotated grid.
            nodes: Nodes coordinates of the rotated grid.

        """
        if rotation_matrix is not None:
            if rotation_matrix.shape != (3, 3):
                raise ValueError("Expected `rotation_matrix` of shape (3, 3).")

        if rotation_matrix is None:
            # Wrap outputs of the `pp.map_geometry.map_grid(sd)`
            cc, fn, fc, rot_mat, dim_bool, nodes = pp.map_geometry.map_grid(
                sd, tol=1.0e-10
            )
            self.sd: pp.Grid = sd
            self.cell_centers: np.ndarray = cc
            self.face_normals: np.ndarray = fn
            self.face_centers: np.ndarray = fc
            self.nodes: np.ndarray = nodes
            self.rotation_matrix: np.ndarray = rot_mat
            self.dim_bool: np.ndarray = dim_bool
            self.dim: int = sum(self.dim_bool)
        else:
            self.sd = sd

            # use the provided rotation matrix
            self.rotation_matrix = rotation_matrix

            full_nodes = rotation_matrix @ sd.nodes
            full_cell_centers = rotation_matrix @ sd.cell_centers
            full_face_centers = rotation_matrix @ sd.face_centers
            full_face_normals = rotation_matrix @ sd.face_normals

            spans = np.ptp(full_nodes, axis=1)
            dim_bool = spans > tol
            self.nodes = full_nodes[dim_bool, :]
            self.cell_centers = full_cell_centers[dim_bool, :]
            self.face_centers = full_face_centers[dim_bool, :]
            self.face_normals = full_face_normals[dim_bool, :]
            self.dim_bool = dim_bool
            self.dim = int(dim_bool.sum())

    def __str__(self):
        return "Rotated pseudo-grid object."

    def __repr__(self):
        return (
            "Rotated pseudo-grid object with attributes:\n"
            + "cell_centers\n"
            + "face_normals\n"
            + "face_centers\n"
            + "rotation_matrix\n"
            + "dim\n"
            + "dim_bool\n"
            + "nodes"
        )


def build_canonical_frames(mdg: pp.MixedDimensionalGrid) -> None:
    """Populate _CANONICAL_FRAMES with (rot_matrix, dim_bool) for each grid/intf in mdg."""

    _CANONICAL_FRAMES.clear()
    _ROTATED_CACHE.clear()

    # (i) Top-dimensional subdomains: identity
    for sd in mdg.subdomains(dim=mdg.dim_max()):
        _CANONICAL_FRAMES[_key(sd)] = (
            np.eye(3),
            np.array([True, True, True], dtype=bool),
        )

    # (ii) 0D subdomains/interfaces: no rotation
    for sd in mdg.subdomains(dim=0):
        _CANONICAL_FRAMES[_key(sd)] = (
            None,
            np.array([False, False, False], dtype=bool),
        )
    for intf in mdg.interfaces(dim=0):
        _CANONICAL_FRAMES[_key(intf)] = (
            None,
            np.array([False, False, False], dtype=bool),
        )

    # (iii) 1D and 2D subdomains: use their own mapped geometry
    for dim in (1, 2):
        for sd in mdg.subdomains(dim=dim):
            sd_rot = mdamr.RotatedGrid(sd)
            rot_matrix = sd_rot.rotation_matrix
            dim_bool = sd_rot.dim_bool
            _CANONICAL_FRAMES[_key(sd)] = (rot_matrix, np.array(dim_bool, dtype=bool))

    # (iv) 1D and 2D interfaces: inherit from lower-dimensional neighbour
    for dim in (1, 2):
        for intf in mdg.interfaces(dim=dim):
            sd_high, sd_low = mdg.interface_to_subdomain_pair(intf)
            assert intf.dim == sd_low.dim
            rot_matrix, dim_bool = _CANONICAL_FRAMES[_key(sd_low)]  # type:ignore
            _CANONICAL_FRAMES[_key(intf)] = (rot_matrix, dim_bool)


def canonical_frame(g: GridLike) -> Tuple[np.ndarray | None, np.ndarray, int]:
    """Return (rotation_matrix, dim_bool, dim) for any subdomain or interface.

    rotation_matrix may be None for 0D objects.
    dim_bool is a length-3 bool array marking active coordinates.
    dim is the effective dimension (sum(dim_bool)).
    """
    k = _key(g)
    if k not in _CANONICAL_FRAMES:
        raise KeyError(
            f"No canonical frame registered for {k}. "
            "Call build_canonical_frames(mdg) first."
        )
    rot_matrix, dim_bool = _CANONICAL_FRAMES[k]
    dim_bool = np.array(dim_bool, dtype=bool)
    dim = int(dim_bool.sum())
    return rot_matrix, dim_bool, dim


def rotate_grid(g: GridLike) -> mdamr.RotatedGrid:
    """Return the canonically rotated version of g, cached by g-type + g.id.

    Requires build_canonical_frames(mdg) to have been called beforehand.
    """
    k = _key(g)

    # Cached?
    if k in _ROTATED_CACHE:
        return _ROTATED_CACHE[k]

    # Get canonical frame
    if k not in _CANONICAL_FRAMES:
        raise KeyError(
            f"No canonical rotation registered for grid/interface {k}. "
            "Call build_canonical_frames(mdg) first."
        )

    rot_matrix, dim_bool = _CANONICAL_FRAMES[k]

    if rot_matrix is None:
        # For 0D grids we don't need a RotatedGrid
        raise ValueError(
            "Zero-dimensional grids/interfaces have no canonical rotation."
        )

    g_rot = mdamr.RotatedGrid(g, rotation_matrix=rot_matrix)  # type:ignore
    _ROTATED_CACHE[k] = g_rot
    return g_rot
