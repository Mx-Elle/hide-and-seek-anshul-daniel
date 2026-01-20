# written by daniel
import collections
import logging
import numpy as np
from navigator import Navigator, navmesh_to_arrays
from agent_base import Agent
from Mesh.nav_mesh import NavMesh
from world_state import WorldState


class HiderAgent(Agent):
    def __init__(self, world_map: NavMesh, max_speed: float = 5.0):
        super().__init__(world_map, max_speed)
        self.name = "theHider"
        self.world_map = world_map

        self.nav = Navigator()
        verts, tris, nbrs = navmesh_to_arrays(world_map)
        self.nav.load_mesh(verts, tris, nbrs)

        self.dist_matrix = self.nav.get_distance_matrix()
        self.centroids = self.nav.get_all_centroids()
        self.num_nodes = self.dist_matrix.shape[0]

        self.adjacency = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for i, neighbors in enumerate(nbrs):
            count = 0
            for n in neighbors:
                if n >= 0:
                    self.adjacency[i, n] = 1.0
                    count += 1
            self.adjacency[i, i] = 1.0
            if count > 0:
                self.adjacency[i] /= count + 1

        self.visit_history = collections.deque(maxlen=6)
        self.current_path = []
        self.path_index = 0
        self.committed_target_pos = None

        self.debug_path = []
        self.debug_target = None

    @property
    def is_seeker(self) -> bool:
        return False

    def _update_visit(self, current_node_idx: int):
        # Keep track of visited areas to avoid staying in the same spot.
        if current_node_idx >= 0 and (
            not self.visit_history or self.visit_history[-1] != current_node_idx
        ):
            self.visit_history.append(current_node_idx)

    def act(self, state: WorldState) -> tuple[float, float] | None:
        me = state.hider_position
        if me is None:
            return None

        my_pos = (me.x, me.y)
        curr_node_idx = self.nav.find_triangle(my_pos[0], my_pos[1])
        if curr_node_idx == -1:
            dists = np.linalg.norm(self.centroids - np.array(my_pos), axis=1)
            curr_node_idx = int(np.argmin(dists))

        self._update_visit(curr_node_idx)

        target_wp = None
        seeker_pos = None
        if state.seeker_position is not None:
            seeker_pos = np.array([state.seeker_position.x, state.seeker_position.y])

        # ESCAPE LOGIC
        if seeker_pos is not None:
            # Pick a node that maximizes distance from seeker
            dists_to_seeker = np.linalg.norm(self.centroids - seeker_pos, axis=1)
            far_mask = dists_to_seeker > 50.0
            scores = dists_to_seeker.copy()

            # Penalize recently visited nodes
            for visited_idx in self.visit_history:
                scores[visited_idx] *= 0.5

            # Mask out nodes too close to seeker
            scores[~far_mask] = -1.0
            scores[curr_node_idx] = -1.0  # avoid staying in place

            best_idx = int(np.argmax(scores))
            target_pos = self.centroids[best_idx]

            # Commit to a path to escape
            self.committed_target_pos = target_pos
            self.debug_target = target_pos

            self.current_path = self.nav.find_path(
                my_pos[0], my_pos[1], target_pos[0], target_pos[1]
            )
            self.path_index = 1
            self.debug_path = self.current_path

        # EXPLORE if no seeker detected
        elif (
            self.committed_target_pos is None
            or not self.current_path
            or self.path_index >= len(self.current_path)
        ):
            # Pick a random far node to explore
            unvisited_nodes = list(set(range(self.num_nodes)) - set(self.visit_history))
            if not unvisited_nodes:
                unvisited_nodes = list(range(self.num_nodes))
            target_idx = np.random.choice(unvisited_nodes)
            target_pos = self.centroids[target_idx]

            self.committed_target_pos = target_pos
            self.debug_target = target_pos

            self.current_path = self.nav.find_path(
                my_pos[0], my_pos[1], target_pos[0], target_pos[1]
            )
            self.path_index = 1
            self.debug_path = self.current_path

        # FOLLOW PATH
        if target_wp is None and self.current_path and len(self.current_path) > 1:
            while self.path_index < len(self.current_path):
                wp = self.current_path[self.path_index]
                d = np.hypot(wp[0] - my_pos[0], wp[1] - my_pos[1])
                is_final = self.path_index == len(self.current_path) - 1

                tolerance = 2.0 if not is_final else 0.5

                if d < tolerance:
                    self.path_index += 1
                else:
                    target_wp = wp
                    break

        if target_wp is None:
            return (0.0, 0.0)

        dx = target_wp[0] - my_pos[0]
        dy = target_wp[1] - my_pos[1]
        dist = np.hypot(dx, dy)

        if dist < 0.1:
            return (0.0, 0.0)

        safe_max_speed = self.max_speed * 0.99
        scale = safe_max_speed / dist
        if dist < safe_max_speed:
            scale = 1.0

        return (dx * scale, dy * scale)
