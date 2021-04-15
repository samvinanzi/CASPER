"""
    Implementation of the A* algorithm for the Agent.
"""

from .astar import AStar
import math
import numpy as np


class RoboAStar(AStar):
    def __init__(self, robot, map):
        AStar.__init__(self)
        self.robot = robot
        self.map = map

    def heuristic_cost_estimate(self, current, goal):
        """ Chebyshev distance. """
        #current_coords = current.data
        #goal_coords = goal.data
        dx = abs(current[0] - goal[0])
        dy = abs(current[1] - goal[1])
        return max(dx, dy)

    def distance_between(self, n1, n2):
        return 1

    def calculate_coordinate(self, direction, delta=0.4):
        assert direction in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"], "Invalid direction"
        delta_diagonal = math.sqrt(2) / 2 * delta
        x = 0
        z = 0
        if direction == "N":
            z += delta
        elif direction == "E":
            x -= delta
        elif direction == "S":
            z -= delta
        elif direction == "W":
            x += delta
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
        return np.array([x, 0, z], dtype=float)

    def global_from_local(self, p_local):
        robot = self.robot.getSelf()
        R = np.array(robot.getOrientation())
        R = R.reshape(3, 3)
        T = np.array(robot.getPosition())
        p_global = np.dot(R, p_local) + T
        return (p_global[0], p_global[2])

    def is_valid(self, destination):
        return self.map.in_room(destination)

    def is_occupied(self, destination):
        return self.map.on_obstacle(destination)

    def neighbors(self, node):
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        neighbors = []
        for direction in directions:
            new_position = self.global_from_local(self.calculate_coordinate(direction))
            if self.is_valid(new_position) and not self.is_occupied(new_position):
                #new_node = AStar.SearchNode(new_position)   # todo initial gscore and fscore?
                #neighbors.append(new_node)
                neighbors.append(new_position)
        return neighbors

    def is_goal_reached(self, current, goal, radius=0.2):
        dist = math.hypot(current[0] - goal[0], current[1] - goal[1])
        return dist <= radius
