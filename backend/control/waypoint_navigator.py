import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


Vector2 = Tuple[float, float]


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float


@dataclass(frozen=True)
class NavigationCommand:
    """Normalized world-frame acceleration request for simulation/testing."""

    acceleration_x: float
    acceleration_y: float
    waypoint_index: Optional[int]
    complete: bool


class WaypointNavigator:
    """Planar waypoint controller; deliberately independent of RC protocols."""

    def __init__(
        self,
        waypoints: Iterable[Waypoint],
        position_gain: float = 0.8,
        velocity_gain: float = 2.0,
        max_speed: float = 1.0,
        max_acceleration: float = 1.0,
        acceptance_radius: float = 0.15,
        arrival_speed: float = 0.1,
    ) -> None:
        self.waypoints = tuple(waypoints)
        self.position_gain = position_gain
        self.velocity_gain = velocity_gain
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.acceptance_radius = acceptance_radius
        self.arrival_speed = arrival_speed
        self.waypoint_index = 0

        positive_values = (
            position_gain,
            velocity_gain,
            max_speed,
            max_acceleration,
            acceptance_radius,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive_values):
            raise ValueError("controller gains and limits must be finite and positive")
        if not math.isfinite(arrival_speed) or arrival_speed < 0:
            raise ValueError("arrival_speed must be finite and non-negative")
        if any(not math.isfinite(point.x) or not math.isfinite(point.y) for point in self.waypoints):
            raise ValueError("waypoint coordinates must be finite")

    def update(self, position: Vector2, velocity: Vector2) -> NavigationCommand:
        position = self._vector(position, "position")
        velocity = self._vector(velocity, "velocity")

        if self.waypoint_index >= len(self.waypoints):
            return self._neutral_command()

        waypoint = self.waypoints[self.waypoint_index]
        error = (waypoint.x - position[0], waypoint.y - position[1])
        distance = math.hypot(*error)
        speed = math.hypot(*velocity)

        if distance <= self.acceptance_radius and speed <= self.arrival_speed:
            self.waypoint_index += 1
            if self.waypoint_index >= len(self.waypoints):
                return self._neutral_command()
            return self._command(0.0, 0.0)

        desired_velocity = self._limit_vector(
            (error[0] * self.position_gain, error[1] * self.position_gain),
            self.max_speed,
        )
        acceleration = self._limit_vector(
            (
                (desired_velocity[0] - velocity[0]) * self.velocity_gain,
                (desired_velocity[1] - velocity[1]) * self.velocity_gain,
            ),
            self.max_acceleration,
        )
        return self._command(
            acceleration[0] / self.max_acceleration,
            acceleration[1] / self.max_acceleration,
        )

    def _command(self, acceleration_x: float, acceleration_y: float) -> NavigationCommand:
        return NavigationCommand(
            acceleration_x=acceleration_x,
            acceleration_y=acceleration_y,
            waypoint_index=self.waypoint_index,
            complete=False,
        )

    def _neutral_command(self) -> NavigationCommand:
        return NavigationCommand(0.0, 0.0, None, True)

    @staticmethod
    def _vector(value: Vector2, name: str) -> Vector2:
        if len(value) != 2 or any(not math.isfinite(component) for component in value):
            raise ValueError(f"{name} must contain two finite values")
        return float(value[0]), float(value[1])

    @staticmethod
    def _limit_vector(vector: Vector2, limit: float) -> Vector2:
        magnitude = math.hypot(*vector)
        if magnitude <= limit:
            return vector
        scale = limit / magnitude
        return vector[0] * scale, vector[1] * scale