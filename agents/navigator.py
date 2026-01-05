import numpy as np
import shapely
from Mesh.nav_mesh import NavMesh
import nav_rs

POINT_DTYPE = np.dtype([("x", np.float32), ("y", np.float32)])
NODE_DTYPE = np.dtype(
    [("centroid", POINT_DTYPE), ("edge_start", np.uint64), ("edge_count", np.uint64)]
)
EDGE_DTYPE = np.dtype(
    [
        ("to", np.uint64),
        ("cost", np.uint32),
        ("left", POINT_DTYPE),
        ("right", POINT_DTYPE),
        ("padding", np.uint32),
    ]
)


class Navigator:
    def __init__(self, mesh: NavMesh, agent_radius: float = 7.0):
        self.mesh = mesh
        self.agent_radius = agent_radius
        self.cells = sorted(mesh.cells, key=lambda c: c._id)
        self.cell_to_idx = {cell: i for i, cell in enumerate(self.cells)}

        nlen = len(self.cells)
        elen = sum(len(cell.neighbors) for cell in self.cells)

        self.nodes = np.zeros(nlen, dtype=NODE_DTYPE)
        self.edges = np.zeros(elen, dtype=EDGE_DTYPE)
        self._walkable_area = None

        edge_idx = 0
        for i, cell in enumerate(self.cells):
            centroid = cell.polygon.centroid
            self.nodes[i]["centroid"] = (centroid.x, centroid.y)
            self.nodes[i]["edge_start"] = edge_idx
            self.nodes[i]["edge_count"] = len(cell.neighbors)

            for neighbor, portal in cell.neighbors.items():
                self.edges[edge_idx]["to"] = self.cell_to_idx[neighbor]
                self.edges[edge_idx]["cost"] = int(cell.distance(neighbor))

                p1, p2 = portal.coords[0], portal.coords[1]
                dx = neighbor.polygon.centroid.x - centroid.x
                dy = neighbor.polygon.centroid.y - centroid.y
                cross = dx * (p1[1] - centroid.y) - dy * (p1[0] - centroid.x)

                left, right = (p1, p2) if cross > 0 else (p2, p1)
                self.edges[edge_idx]["left"] = (left[0], left[1])
                self.edges[edge_idx]["right"] = (right[0], right[1])
                edge_idx += 1

    @property
    def walkable_area(self):
        if self._walkable_area is None:
            self._walkable_area = shapely.unary_union([c.polygon for c in self.cells])
        return self._walkable_area

    def find_path(
        self, start: shapely.Point, target: shapely.Point
    ) -> list[shapely.Point]:
        start_cell = self.mesh.find_cell(start)
        target_cell = self.mesh.find_cell(target)

        if not start_cell or not target_cell:
            return []

        path_points = nav_rs.find_path(
            self.nodes,
            self.edges,
            self.cell_to_idx[start_cell],
            self.cell_to_idx[target_cell],
            nav_rs.Point(target.x, target.y),
        )

        return [shapely.Point(p.x, p.y) for p in path_points]

    def optimize_path_from_position(
        self, pos: shapely.Point, waypoints: list[shapely.Point]
    ) -> list[shapely.Point]:
        if len(waypoints) <= 1:
            return waypoints

        farthest = 0
        for i, wp in enumerate(waypoints):
            thick_line = shapely.LineString([pos, wp]).buffer(
                self.agent_radius, cap_style="flat"
            )
            if not self.walkable_area.contains(thick_line):
                break
            farthest = i

        return waypoints[farthest:]
