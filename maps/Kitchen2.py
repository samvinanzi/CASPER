"""
    Obstacles in the kitchen environment. Version 2.
"""

from shapely.geometry import Polygon
from .Map import Map


class Kitchen2(Map):
    def __init__(self):
        Map.__init__(self)

        table = Polygon([[0.98, 0.2], [1.88, 0.2], [1.88, 1.09], [0.98, 1.09]])
        coffee_table = Polygon([[3.2, 1.44], [3.8, 1.44], [3.8, 2.14], [3.2, 2.14]])
        surfaces = Polygon([[-1.62, 2.14], [-0.95, 2.14], [-0.95, -1.42], [1.95, -1.42], [1.95, -2.15], [-1.62, -2.15]])

        self.obstacles = [table, coffee_table, surfaces]
        self.room = Polygon([[-1.62, -2.15], [-1.62, 2.14], [3.80, 2.14], [3.80, -1.13], [1.95, -1.13], [1.95, -2.15]])
