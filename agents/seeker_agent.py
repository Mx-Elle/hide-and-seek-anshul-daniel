# Made By Anshul Noori
import collections

import numpy as np
from navigator import Navigator, navmesh_to_arrays

from agent_base import Agent
from Mesh.nav_mesh import NavMesh
from world_state import WorldState


class SeekerAgent(Agent):
    def __init__(
        self,
        world_map: NavMesh,
        max_speed: float = 5.0,
    ):
        super().__init__(world_map, max_speed)
        self.name = "Predator Seeker"
        self.world_map = world_map

        self.nav = Navigator()
        verts, tris, nbrs = navmesh_to_arrays(world_map)
        self.nav.load_mesh(verts, tris, nbrs)

        self.dist_matrix = self.nav.get_distance_matrix()
        self.centroids = self.nav.get_all_centroids()
        self.num_nodes = self.dist_matrix.shape[0]

        self.belief = np.ones(self.num_nodes, dtype=np.float32) / self.num_nodes

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

        self.last_seen_pos = None
        self.last_seen_vel = np.array([0.0, 0.0])
        self.investigating = False

        self.debug_path = []
        self.debug_target = None

    @property
    def is_seeker(self) -> bool:
        return True

    def _update_belief(self, current_node_idx: int):
        if current_node_idx >= 0:
            self.belief[current_node_idx] = 0.0
            if not self.visit_history or self.visit_history[-1] != current_node_idx:
                self.visit_history.append(current_node_idx)

        diffusion_rate = 0.05
        diffused_belief = self.adjacency.T @ self.belief
        self.belief = (1.0 - diffusion_rate) * self.belief + (
            diffusion_rate * diffused_belief
        )

        total = np.sum(self.belief)
        if total > 0:
            self.belief /= total
        else:
            self.belief[:] = 1.0 / self.num_nodes

    def act(self, state: WorldState) -> tuple[float, float] | None:
        me = state.seeker_position
        if me is None:
            return None

        my_pos = (me.x, me.y)

        curr_node_idx = self.nav.find_triangle(my_pos[0], my_pos[1])
        if curr_node_idx == -1:
            dists = np.linalg.norm(self.centroids - np.array(my_pos), axis=1)
            curr_node_idx = int(np.argmin(dists))

        self._update_belief(curr_node_idx)

        target_wp = None

        # HUNT
        if state.hider_position is not None:
            self.investigating = False
            self.committed_target_pos = None

            h_pos = np.array([state.hider_position.x, state.hider_position.y])

            if self.last_seen_pos is not None:
                self.last_seen_vel = h_pos - self.last_seen_pos
            self.last_seen_pos = h_pos

            dist_to_hider = np.linalg.norm(h_pos - np.array(my_pos))
            prediction_time = min(dist_to_hider / self.max_speed, 1.5)

            target_pos = h_pos + (self.last_seen_vel * prediction_time * 30.0)
            self.debug_target = target_pos

            path = self.nav.find_path(
                my_pos[0], my_pos[1], target_pos[0], target_pos[1]
            )
            self.debug_path = path

            if path and len(path) > 1:
                target_wp = path[1]
            elif path:
                target_wp = path[-1]

        # INVESTIGATE
        elif self.last_seen_pos is not None:
            dist_to_prediction = 1000.0
            if self.committed_target_pos is not None:
                dist_to_prediction = np.linalg.norm(
                    np.array(self.committed_target_pos) - np.array(my_pos)
                )

            if self.investigating and dist_to_prediction < 3.0:
                self.last_seen_pos = None
                self.investigating = False
                self.committed_target_pos = None  # Reset commitment
                self.current_path = []

            else:
                if not self.investigating:
                    self.investigating = True

                    target_pos = self.last_seen_pos + (self.last_seen_vel * 60.0)

                    self.committed_target_pos = target_pos  # COMMIT
                    self.debug_target = target_pos

                    self.current_path = self.nav.find_path(
                        my_pos[0], my_pos[1], target_pos[0], target_pos[1]
                    )
                    self.path_index = 1
                    self.debug_path = self.current_path

        # EXPLORE
        if target_wp is None and self.last_seen_pos is None:
            if self.committed_target_pos is not None:
                dist_to_commit = np.linalg.norm(
                    np.array(self.committed_target_pos) - np.array(my_pos)
                )
                if dist_to_commit < 5.0:
                    self.committed_target_pos = None

            if (
                self.committed_target_pos is None
                or not self.current_path
                or self.path_index >= len(self.current_path)
            ):
                current_frame = getattr(state, "frame", 0)
                is_late_game = current_frame > 3600

                dists = self.dist_matrix[curr_node_idx]

                if is_late_game:
                    scores = dists.copy()
                else:
                    scores = self.belief / (dists + 50.0)

                    far_mask = dists > 150.0
                    if np.any(scores[far_mask] > 0):
                        scores[~far_mask] = -1.0

                scores[curr_node_idx] = -1.0
                for visited_idx in self.visit_history:
                    scores[visited_idx] *= 0.01

                best_idx = int(np.argmax(scores))
                target_pos = self.centroids[best_idx]

                self.committed_target_pos = target_pos
                self.debug_target = target_pos

                self.current_path = self.nav.find_path(
                    my_pos[0], my_pos[1], target_pos[0], target_pos[1]
                )
                self.path_index = 1
                self.debug_path = self.current_path

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
