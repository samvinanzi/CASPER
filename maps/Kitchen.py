"""
    Obstacles in the kitchen environment
"""

from shapely.geometry import Polygon
from .Map import Map


class Kitchen(Map):
    def __init__(self):
        Map.__init__(self)

        table = Polygon([[0.53, 0.16], [2.33, 0.16], [2.33, 1.14], [0.53, 1.14]])
        surfaces = Polygon([[-1.62, 2.14], [-0.95, 2.14], [-0.95, -1.42], [1.95, -1.42], [1.95, -2.15], [-1.62, -2.15]])

        self.obstacles = [table, surfaces]
        self.room = Polygon([[-1.62, -2.15], [-1.62, 2.14], [3.80, 2.14], [3.80, -1.13], [1.95, -1.13], [1.95, -2.15]])
