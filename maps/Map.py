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
        self.room = None    # Outer shape of the environment

    @staticmethod
    def does_intersect(poly1: Polygon, poly2: Polygon):
        """
        Checks if two polygons intersect.

        :param poly1: Polygon
        :param poly2: Polygon
        :return: True/False
        """
        return poly1.intersection(poly2).area > 0

    @staticmethod
    def coord_to_point(coordinates):
        """
        Transforms a tuple representing a (x,y) coordinate in a shapely.Point data structure.

        :param coordinates: tuple
        :return: Point
        """
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

    def distance_to_obstacles(self, coordinate):
        """
        Returns the minimum distance from the specified coordinate to the obstacles in the map, or infinite if the point
        is outside of the environment.

        :param coordinate: (x,y)
        :return: float distance
        """
        if not self.in_room(coordinate):
            return float("inf")     # If the point is outside the room, distance is infinite
        elif self.on_obstacle(coordinate):
            return 0                # If the point is in or touches an obstacle, distance is zero
        else:
            distances = []
            # Calculate the distance from the point to each obstacle
            for obstacle in self.obstacles:
                distances.append(obstacle.distance(self.coord_to_point(coordinate)))
            # Calculate the distance from the point to the wallks of the room
            distances.append(self.room.exterior.distance(self.coord_to_point(coordinate)))
            return round(min(distances), 2)

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

    def add_point_to_plot(self, point, style="k.", text=None):
        """
        Adds a single point to the plot.

        :param point: (x,y) float tuple
        :param style: Drawing style of the point (see Matplotlib.pyplot documentation)
        :param text: optional label to add to the plot
        :return: None
        """
        plt.plot(point[0], point[1], style)
        if text:
            plt.text(point[0], point[1], text)
