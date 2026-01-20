import numpy as np
from Mesh.nav_mesh import NavMesh


def navmesh_to_arrays(mesh: NavMesh):
    cells = list(mesh.cells)
    cell_to_idx = {cell: i for i, cell in enumerate(cells)}

    all_coords = []
    triangles = []

    coord_to_idx = {}
    for cell in cells:
        poly = cell.polygon
        coords = list(poly.exterior.coords)[:3]
        tri_indices = []
        for c in coords:
            if c not in coord_to_idx:
                coord_to_idx[c] = len(all_coords)
                all_coords.append(c)
            tri_indices.append(coord_to_idx[c])
        triangles.append(tri_indices)
    neighbors = []

    for cell in cells:
        tri_nbrs = [-1, -1, -1]
        poly_coords = list(cell.polygon.exterior.coords)[:3]
        for nbr_cell, border in cell.neighbors.items():
            if nbr_cell not in cell_to_idx:
                continue
            nbr_idx = cell_to_idx[nbr_cell]
            for j in range(3):
                p1 = poly_coords[j]
                p2 = poly_coords[(j + 1) % 3]
                border_coords = list(border.coords)
                if (p1 in border_coords) and (p2 in border_coords):
                    tri_nbrs[j] = nbr_idx
                    break
        neighbors.append(tri_nbrs)

    return (
        np.array(all_coords, dtype=np.float32),
        np.array(triangles, dtype=np.int32),
        np.array(neighbors, dtype=np.int32),
    )
