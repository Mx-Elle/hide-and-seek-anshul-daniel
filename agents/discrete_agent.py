import math
import logging
import shapely
from agent_base import Agent
from world_state import WorldState
from Mesh.nav_mesh import NavMesh
from agents.navigator import Navigator

logger = logging.getLogger("DiscreteAgent")


class DiscreteAgent(Agent):
    def __init__(self, world_map: NavMesh, max_speed: float, agent_radius: float = 7.0):
        Agent.__init__(self, world_map, max_speed)
        self.path = []
        self.current_target = None
        self.nav = Navigator(world_map, agent_radius)

    def act(self, state: WorldState) -> tuple[float, float] | None:
        if not self.map.in_bounds(state.target):
            return None

        if self.current_target != state.target:
            self.current_target = state.target

            if not (
                self.map.find_cell(state.location) and self.map.find_cell(state.target)
            ):
                return None

            self.path = self.nav.find_path(state.location, state.target)

            if not self.path:
                return None

        if not self.path:
            return None

        self.path = self.nav.optimize_path_from_position(state.location, self.path)

        if not self.path:
            return None

        target = self.path[0]
        dx, dy = target.x - state.location.x, target.y - state.location.y
        dist = math.hypot(dx, dy)

        if dist < self.max_speed and len(self.path) > 1:
            self.path.pop(0)
            target = self.path[0]
            dx, dy = target.x - state.location.x, target.y - state.location.y
            dist = math.hypot(dx, dy)

        return (
            (
                min(dist, self.max_speed) * dx / dist,
                min(dist, self.max_speed) * dy / dist,
            )
            if dist > 0
            else (0, 0)
        )
