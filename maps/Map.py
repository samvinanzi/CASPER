"""
    Abstract class for a generic map representation.
"""


from abc import abstractmethod
from shapely.geometry import Polygon, Point
from shapely import speedups
import matplotlib.pyplot as plt
from random import uniform
from datetime import datetime


class Map:
    @abstractmethod
    def __init__(self):
        #if speedups.available:
            #speedups.enable()   # Ensure Shapely speed-ups are enabled, when available -- Starting with Shapely 2.0, equivalent speedups are always available
        self.obstacles = None
        self.prepared_obstacles = None  # Prepared items enable fast computations
        self.room = None    # Outer shape of the environment
        self.prepared_room = None
        self.free_space = None
        self.prepared_free_space = None
        self.axesInverted = False
        self.goalCounter = 0
        self.points = {}

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

    def in_room(self, point):
        """
        Checks if the specified coordinate is inside the room.

        :param point: shapely.Point
        :return: True/False
        """
        return self.prepared_room.contains(point)

    def on_obstacle(self, point):
        """
        Checks if an obstacle contains the specified coordinate.

        :param point: shapely.Point
        :return: True/False
        """
        for prepared_obstacle in self.prepared_obstacles:
            if prepared_obstacle.contains(point):
                return True
        return False

    def in_free_space(self, point):
        """
        Checks if the specified coordinate is inside the free space (avoid calculating 'in_room' and 'not on_obstacles')

        :param point: shapely.Point
        :return: True/False
        """
        return self.prepared_free_space.contains(point)

    def distance_to_obstacles(self, point):
        """
        Returns the minimum distance from the specified coordinate to the obstacles in the map, or infinite if the point
        is outside of the environment.

        :param point: shapely.Point
        :return: float distance
        """
        """
        I decided to skip these checks, since they are typically done before or after this function. This should speed
        it up a notch.
        
        if not self.in_room(point):
            return float("inf")     # If the point is outside the room, distance is infinite
        elif self.on_obstacle(point):
            return 0                # If the point is in or touches an obstacle, distance is zero
        else:
        """
        distances = []
        # Calculate the distance from the point to each obstacle
        for obstacle in self.obstacles:
            distances.append(obstacle.distance(point))
        # Calculate the distance from the point to the walls of the room
        distances.append(self.room.exterior.distance(point))
        return round(min(distances), 2)

    def visualize(self, agentName="Agent"):
        """
        Prints a 2D schematic of the environment.

        :return: None
        """
        # Prepare the plot
        ax = plt.gca()

        if(not self.axesInverted):
            ax.invert_xaxis()  # invert only the X axis
            self.axesInverted = True
        
        # Webots coordinate system has Y going right and X going up, so we swap them for visualization
        plt.xlabel("Y")
        plt.ylabel("X")
        
        # Mark the origin for reference
        #plt.plot(0, 0, 'rP')
        
        # Draw environment boundaries (external walls)
        x, y = self.room.exterior.xy
        plt.plot(y, x, 'k-')
        
        obstaclesColor = ['lightgrey', 'darkgrey', 'dimgray', 'grey']
        i = 0
        # Draw obstacles and fill them
        for obstacle in self.obstacles:
            x, y = obstacle.exterior.xy
            plt.fill(y, x, obstaclesColor[i])
            i += 1

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        #filename = f"../../../path_planning/GeneratedPaths/map_{timestamp}.png"
        filename = f"../../../path_planning/GeneratedPaths/map_{agentName}.png"
            
        plt.savefig(filename)
        #plt.show()

    def add_point_to_plot(self, point, style="k.", text=None):
        """
        Adds a single point to the plot.

        :param point: (x,y) float tuple
        :param style: Drawing style of the point (see Matplotlib.pyplot documentation)
        :param text: optional label to add to the plot
        :return: None
        """
        plt.plot(point[1], point[0], style)# Webots coordinate system has Y going right and X going up, so we swap them for visualization
        if text:
            if(text == 'GOAL'):
                self.goalCounter += 1
                goalsSameCoord = self.add_point_goals(point[1], point[0]) 
                if(goalsSameCoord > 0):
                    num_spaces = 5 + goalsSameCoord + 2 * (goalsSameCoord - 1)
                    space_string = "  " * num_spaces
                    plt.text(point[1], point[0], space_string + ', ' + str(self.goalCounter))
                else:
                    plt.text(point[1], point[0], text + ' ' + str(self.goalCounter))
            else:
                plt.text(point[1], point[0], text)

    def bounding_box(self):
        """
        Returns the two diagonal corners of a bounding box containing the room shape.

        :return: upper left corner, lower right corner
        """
        box = self.room.minimum_rotated_rectangle
        x, y = box.exterior.coords.xy
        p1 = (x[3], y[3])
        p2 = (x[1], y[1])
        return p1, p2

    def sample_point(self):
        """
        Samples a point and returns it if it occupies a valid free space inside the room.
        :return: (x, y) coordinate
        """
        p1, p2 = self.bounding_box()
        approved = False
        coord = (0, 0)
        while not approved:
            x = round(uniform(p1[0], p2[0]), 2)
            y = round(uniform(p1[1], p2[1]), 2)
            coord = (x, y)
            point = self.coord_to_point(coord)
            approved = self.in_room(point) and not self.on_obstacle(point) and self.distance_to_obstacles(point) > 0.2
        return coord

    def sampling_test(self, n=20):
        """
        Debug function to test the validity of the point sampling.
        :param n: int, number of samples
        :return: None
        """
        for i in range(1, n+1):
            p = self.sample_point()
            self.add_point_to_plot(p, text=i)
        self.visualize()

    def add_point_goals(self, x, y):
        coord = (x, y)  # tuples can be dict keys
        if coord in self.points:
            self.points[coord] += 1
        else:
            self.points[coord] = 1
        return self.points[coord] - 1 #do not consider the one just added
