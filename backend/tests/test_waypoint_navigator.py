import unittest
from datetime import datetime

from control.waypoint_navigator import Waypoint, WaypointNavigator


class WaypointNavigatorTests(unittest.TestCase):
    def test_command_is_bounded_and_points_toward_waypoint(self):
        navigator = WaypointNavigator([Waypoint(100.0, 100.0)])

        command = navigator.update((0.0, 0.0), (0.0, 0.0))

        self.assertGreater(command.acceleration_x, 0.0)
        self.assertGreater(command.acceleration_y, 0.0)
        self.assertLessEqual(command.acceleration_x ** 2 + command.acceleration_y ** 2, 1.0 + 1e-12)

    def test_simulated_motion_reaches_all_waypoints(self):
        navigator = WaypointNavigator([Waypoint(2.0, 0.0), Waypoint(2.0, 1.0)])
        position = [0.0, 0.0]
        velocity = [0.0, 0.0]
        timestep = 0.02
        command = None

        for _ in range(3000):
            command = navigator.update(tuple(position), tuple(velocity))
            print(
                f"{datetime.now().astimezone().isoformat(timespec='milliseconds')} "
                f"sim_time={_ * timestep:.2f}s "
                f"waypoint_index={command.waypoint_index} "
                f"position=({position[0]:.3f}, {position[1]:.3f}) "
                f"velocity=({velocity[0]:.3f}, {velocity[1]:.3f}) "
                f"acceleration=({command.acceleration_x:.3f}, {command.acceleration_y:.3f}) "
                f"complete={command.complete}",
                flush=True,
            )
            if command.complete:
                break

            acceleration = (
                command.acceleration_x * navigator.max_acceleration,
                command.acceleration_y * navigator.max_acceleration,
            )
            for axis in range(2):
                velocity[axis] += acceleration[axis] * timestep
                position[axis] += velocity[axis] * timestep

        self.assertIsNotNone(command)
        self.assertTrue(command.complete, f"route did not complete; final state: {position}, {velocity}")
        self.assertLessEqual(navigator.waypoint_index, len(navigator.waypoints))

    def test_empty_route_returns_neutral_complete_command(self):
        navigator = WaypointNavigator([])

        command = navigator.update((0.0, 0.0), (0.0, 0.0))

        self.assertTrue(command.complete)
        self.assertEqual((command.acceleration_x, command.acceleration_y), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()