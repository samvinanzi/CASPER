"""
Various classes that implement data formats.

"""


class Prediction:
    """
    This class models both observations and goals.
    """
    def __init__(self, name, param, probability=0.0, frontier=None):
        assert isinstance(param, dict), "The parameters must be expressed as a dictionary param:value"
        self.name = name
        self.param = param
        self.probability = probability
        self.frontier = frontier

    def is_goal(self):
        return True if self.probability != 0.0 and self.frontier is not None else False

    def __str__(self):
        return "{0} {1}".format(self.name, self.param)