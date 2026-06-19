"""Module containing a collection of grid helper utilities.

Utility functions found here are from the mdamr package by Jhabriel Varela.

"""

import numpy as np
from scipy.spatial import cKDTree


def is_ccw(coords):
    x = np.array([p[0] for p in coords])
    y = np.array([p[1] for p in coords])
    return np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)) > 0


def point_in_triangle(pt, tri):
    (x, y), (x1, y1), (x2, y2) = pt, tri[0], tri[1]
    (x3, y3) = tri[2]
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if denom == 0:
        return False
    a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
    b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
    c = 1 - a - b
    return (0 < a < 1) and (0 < b < 1) and (0 < c < 1)


def merge_close_vertices(verts, cells, tol=1e-8):
    """
    verts: list of (x,y) tuples or array shape (N,2)
    cells: list of [i0,i1,i2] (connectivity using indices into verts)
    Returns:
        new_verts_arr: array shape (2, N_new)
        new_cells: list of [i0,i1,i2] with reindexed vertices
    """
    arr = np.array(verts)  # shape (N,2)
    if arr.size == 0:
        raise ValueError(
            "merge_close_vertices: received empty vertex list"
            " (no intersection triangles)."
        )
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"merge_close_vertices: expected verts to be (N,2),"
            f" got array shape {arr.shape}."
        )

    tree = cKDTree(arr)
    groups = tree.query_ball_tree(tree, r=tol)

    # Find representative for each original index (smallest in its cluster)
    rep = {}
    for i, neigh in enumerate(groups):
        rep[i] = min(neigh)

    # Build mapping from representatives to consecutive new indices
    uniq_reps = sorted(set(rep.values()))
    new_index = {r: idx for idx, r in enumerate(uniq_reps)}

    # Final old->new map
    old_to_new = {i: new_index[rep[i]] for i in range(len(arr))}

    # Build new vertex list (take reps)
    new_verts = arr[uniq_reps, :]  # shape (N_new,2)

    # Remap cells
    remapped = []
    for tri in cells:
        remapped.append([old_to_new[i] for i in tri])

    return new_verts.T, remapped  # return in (2, N_new) form and updated cells

def ear_clip_triangulate(coords, tol=1e-12):
    verts = coords.copy()
    if not is_ccw(verts):
        verts.reverse()
    tris = []
    max_iter = len(verts) ** 2
    it = 0
    while len(verts) > 3 and it < max_iter:
        it += 1
        n = len(verts)
        ear_found = False
        for i in range(n):
            prev = verts[(i - 1) % n]
            curr = verts[i]
            nxt = verts[(i + 1) % n]
            ax, ay = prev
            bx, by = curr
            cx, cy = nxt
            # convexity test
            if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) <= 0:
                continue
            tri = [prev, curr, nxt]
            # no other point inside
            if any(
                point_in_triangle(p, tri)
                for j, p in enumerate(verts)
                if j not in {(i - 1) % n, i, (i + 1) % n}
            ):
                continue
            # clip ear
            tris.append(tri)
            del verts[i]
            ear_found = True
            break
        if not ear_found:
            break  # degeneracy; bail
    if len(verts) == 3:
        tris.append(verts)
    # filter tiny
    filtered = []
    for t in tris:
        area = 0.5 * abs(np.cross(np.subtract(t[1], t[0]), np.subtract(t[2], t[0])))
        if area > tol:
            filtered.append(t)
    return filtered


def ensure_ccw(cells, coords_arr):
    new_cells = []
    for tri in cells:
        i0, i1, i2 = tri
        p0 = coords_arr[:, i0]
        p1 = coords_arr[:, i1]
        p2 = coords_arr[:, i2]
        cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if cross < 0:
            new_cells.append([i0, i2, i1])
        else:
            new_cells.append([i0, i1, i2])
    return new_cells
