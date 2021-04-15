"""
    Abstract class for a generic map representation.
"""


from abc import abstractmethod
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt


class Map:
    @abstractmethod
    def __init__(self):
        self.obstacles = None
        self.room = None

    @staticmethod
    def does_intersect(poly1: Polygon, poly2: Polygon):
        return poly1.intersection(poly2).area > 0

    @staticmethod
    def coord_to_point(coordinates):
        return Point(coordinates[0], coordinates[1])

    def in_room(self, coordinate):
        """
        Checks if the specified coordinate is inside the room.

        :param coordinate: (x,y)
        :return: True/False
        """
        point = self.coord_to_point(coordinate)
        return self.room.contains(point)

    def on_obstacle(self, coordinate):
        """
        Checks if an obstacle contains the specified coordinate.

        :param coordinate: (x,y)
        :return: True/False
        """
        point = self.coord_to_point(coordinate)
        for obstacle in self.obstacles:
            if obstacle.contains(point):
                return True
        return False

    def visualize(self):
        """
        Prints a 2D schematic of the environment.

        :return: None
        """
        # Prepares the plot
        plt.gca().invert_yaxis()
        plt.xlabel("X")
        plt.ylabel("Z")
        # Marks the origin, for reference
        plt.plot(0, 0, 'rP')
        # Draws the boundaries of the environment (external walls)
        x, y = self.room.exterior.xy
        plt.plot(x, y, 'k-')
        # Draws the obstacles and fills them
        for obstacle in self.obstacles:
            x, y = obstacle.exterior.xy
            plt.fill(x, y)
        plt.show()
