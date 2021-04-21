"""
    Implementation of the A* algorithm for the Agent.
"""

from .astar import AStar
import math
import numpy as np


class RoboAStar(AStar):
    def __init__(self, robot, map, delta=0.4, min_distance=0.2, goal_radius=0.6):
        """
        Initializes the path planning algorithm.

        :param robot: Instance of the robot which is going to be moving.
        :param map: Instance of the map of the room.
        :param delta: Distance [m] that will be traversed at each step.
        :param min_distance: Minimum distance [m] that has to be kept with obstacles and walls.
        :param goal_radius: Distance [m] from the goal which is considered acceptable.
        """
        AStar.__init__(self)
        self.robot = robot
        self.map = map
        self.delta = delta
        self.min_distance = min_distance
        self.goal_radius = goal_radius

    def heuristic_cost_estimate(self, current, goal, D=1, D2=math.sqrt(2)):
        """ Octile distance. """
        dx = abs(current[0] - goal[0])
        dy = abs(current[1] - goal[1])
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
        #return max(dx, dy)     # Chebyshev

    def distance_between(self, n1, n2):
        return 1

    def calculate_coordinate(self, direction):
        assert direction in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"], "Invalid direction"
        delta_diagonal = math.sqrt(2) / 2 * self.delta
        x = 0
        z = 0
        if direction == "N":
            z += self.delta
        elif direction == "E":
            x -= self.delta
        elif direction == "S":
            z -= self.delta
        elif direction == "W":
            x += self.delta
        elif direction == "NE":
            x -= delta_diagonal
            z += delta_diagonal
        elif direction == "SE":
            x -= delta_diagonal
            z -= delta_diagonal
        elif direction == "SW":
            x += delta_diagonal
            z -= delta_diagonal
        elif direction == "NW":
            x += delta_diagonal
            z += delta_diagonal
        x = round(x, 2)
        z = round(z, 2)
        return np.array([x, 0, z], dtype=float)

    def global_from_local(self, current_position, p_local):
        robot = self.robot.getSelf()
        R = np.array(robot.getOrientation())
        R = R.reshape(3, 3)
        T = np.array([current_position[0], 1.27, current_position[1]])
        p_global = np.dot(R, p_local) + T
        return (round(p_global[0], 2), round(p_global[2], 2))

    def is_valid(self, destination):
        return self.map.in_room(destination)

    def is_occupied(self, destination):
        return self.map.on_obstacle(destination)

    def neighbors(self, node):
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        neighbors = []
        for direction in directions:
            new_position = self.global_from_local(node, self.calculate_coordinate(direction))
            distance_from_obstacles = self.map.distance_to_obstacles(new_position)
            # The position must be inside the environment, not occupied by an obstacle and not too close to one of them
            if self.is_valid(new_position) and not self.is_occupied(new_position) and \
                    distance_from_obstacles >= self.min_distance:
                neighbors.append(new_position)
        return neighbors

    def is_goal_reached(self, current, goal):
        dist = math.hypot(current[0] - goal[0], current[1] - goal[1])
        return dist <= self.goal_radius

    def add_text(self, point, style="k.", text=None):
        self.map.add_point_to_plot(point, style, text)
