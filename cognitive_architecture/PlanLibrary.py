"""

These classes model the Plan Library to perform probabilistic goal recognition.

"""

import copy
from anytree import NodeMixin, RenderTree, PreOrderIter
from anytree.exporter import UniqueDotExporter
from enum import Enum


class Marker(Enum):
    """
    The three states in which a node can be ad any time.
    """
    OBSERVED = 1
    UNOBSERVED = 2
    MISSED = 3


# Generic Node

class PLNode(NodeMixin):
    """
    Basic node structure. Inherits from a class of anytree.
    """
    def __init__(self, action, id=0, parent=None, children=None):
        super().__init__()
        self.id = id
        self.name = action
        self.parameters = None
        self.marker = Marker.UNOBSERVED
        self.nt = True  # Non-terminal node?
        self.goal = False   # Goal node?
        self.parent = parent
        if children:
            self.children = children

    def is_observed(self):
        return True if self.marker is Marker.OBSERVED else False

    def is_unobserved(self):
        return True if self.marker is Marker.UNOBSERVED else False

    def is_missed(self):
        return True if self.marker is Marker.MISSED else False

    def mark(self, mark):
        """
        Mark the node in a certain way.

        @param mark: One of the enum Marker values
        @return: None
        """
        assert isinstance(mark, Marker), "Invalid mark provided."
        self.marker = mark

    def __str__(self):
        return "{0} with parameters {1}".format(self.name, self.parameters)


# Terminal nodes

class PnPNode(PLNode):
    def __init__(self, id=0, parent=None, children=None):
        super().__init__(action="Pick&Place", id=id, parent=parent, children=children)
        self.nt = False
        self.parameters = {
            "item": None,
            "destination": None
        }


class EatNode(PLNode):
    def __init__(self, id=0, parent=None, children=None):
        super().__init__(action="Eat", id=id, parent=parent, children=children)
        self.nt = False
        self.parameters = {
            "food": None,
            "vessel": "plate"
        }


class SipNode(PLNode):
    def __init__(self, id=0, parent=None, children=None):
        super().__init__(action="Sip", id=id, parent=parent, children=children)
        self.nt = False
        self.parameters = {
            "beverage": None,
            "vessel": "glass"
        }


class WashNode(PLNode):
    def __init__(self, id=0, parent=None, children=None):
        super().__init__(action="Wash", id=id, parent=parent, children=children)
        self.nt = False
        self.parameters = {
            "item": None,
            "appliance": "sink"
        }


class CookNode(PLNode):
    def __init__(self, id=0, parent=None, children=None):
        super().__init__(action="Cook", id=id, parent=parent, children=children)
        self.nt = False
        self.parameters = {
            "food": None,
            "appliance": "hobs"
        }


# Non-terminal nodes

class PrepareMealNode(PLNode):
    def __init__(self, id=0, parent=None, children=None):
        super().__init__(action="PrepareMeal", id=id, parent=parent, children=children)
        self.parameters = {
            "meal": None,
            "appliance": "hobs",
            "vessel": "plate"
        }
        self.children = [WarmNode(), PnPNode()]


class WarmNode(PLNode):
    def __init__(self, id=0, parent=None, children=None):
        super().__init__(action="Warm", id=id, parent=parent, children=children)
        self.parameters = {
            "food": None,
            "appliance": "hobs"
        }
        self.children = [PnPNode(), CookNode()]


class CleanNode(PLNode):
    def __init__(self, id=0, parent=None, children=None):
        super().__init__(action="Clean", id=id, parent=parent, children=children)
        self.parameters = {
            "item": None,
            "appliance": "sink"
        }
        self.children = [PnPNode(), WashNode()]


# Goal nodes

class BreakfastNode(PLNode):
    def __init__(self, id=0, parent=None, children=None):
        super().__init__(action="Breakfast", id=id, parent=parent, children=children)
        self.goal = True
        self.parameters = {
            "food": None,
            "vessel": "plate"
        }
        self.children = [PnPNode(), EatNode(), CleanNode()]


class LunchNode(PLNode):
    def __init__(self, id=0, parent=None, children=None):
        super().__init__(action="Lunch", id=id, parent=parent, children=children)
        self.goal = True
        self.parameters = {
            "food": None,
            "appliance": "hobs",
            "vessel": "plate"
        }
        self.children = [PrepareMealNode(), EatNode(), CleanNode()]


class DrinkNode(PLNode):
    def __init__(self, id=0, parent=None, children=None):
        super().__init__(action="Drink", id=id, parent=parent, children=children)
        self.goal = True
        self.parameters = {
            "beverage": None,
            "vessel": "glass"
        }
        self.children = [PnPNode(), SipNode(), CleanNode()]


# Plan library

class Plan:
    """
    Represents a plan, that is a tree that models a goal plan.
    """
    def __init__(self, root):
        """
        Constructor.

        @param root: The root node of the tree, representing the goal.
        """
        self.root = root
        self.leaves = list(self.root.leaves)
        self.score = 0
        self.equivalencies = None

    def find_action_index(self, action):
        """
        Searches for the specified node name in the leaves of the plan and returns the indices of the matches.

        @param action: The node name to look for
        @return: List of indices
        """
        indices = [i for i, x in enumerate([l.name for l in self.leaves]) if x == action]
        return indices

    def get_node_by_id(self, id):
        """
        Returns a node, given its id.

        @param id: int id
        @return: The desired node, or None if not found.
        """
        nodes = [node for node in PreOrderIter(self.root)]
        for node in nodes:
            if node.id == id:
                return node
        return None

    def calculate_score(self):
        """
        Calculates the score of the tree as: % of observed nodes * (1 - % of missed nodes).
        This score follows Occam's Razor principle, favoring the simplest explanation and penalizing explanations that
        contain too many missed nodes.

        @return: None
        """
        frontier = self.root.leaves
        completed = 0
        missed = 0
        for leaf in frontier:
            if leaf.is_observed():
                completed += 1
            elif leaf.is_missed():
                missed += 1
        completed = completed / len(frontier)
        missed = missed / len(frontier)
        self.score = completed * (1 - missed)

    def enforce_equivalencies(self):
        """
        Using the equivalency table, checks that the specified parameters are equal. If some are blank, it fills them
        in. If they are conflicting, returns an error message.

        @return: None
        """
        for couple in self.equivalencies:
            nodeA_id = couple[0][0]
            paramA = couple[0][1]
            nodeB_id = couple[1][0]
            paramB = couple[1][1]
            nodeA = self.get_node_by_id(nodeA_id)
            nodeB = self.get_node_by_id(nodeB_id)
            if nodeA.parameters[paramA] != nodeB.parameters[paramB]:
                if nodeA.parameters[paramA] is None:
                    nodeA.parameters[paramA] = nodeB.parameters[paramB]
                elif nodeB.parameters[paramB] is None:
                    nodeB.parameters[paramB] = nodeA.parameters[paramA]
                else:
                    print("There is a conflict!\nNode {0}, parameter {1} = {2}\nNode {0}, parameter {1} = {2}".format(
                        nodeA.name, paramA, nodeA.parameters[paramA], nodeB.name, paramB, nodeB.parameters[paramB]
                    ))

    def render(self):
        """
        Render the tree in the terminal.

        @return: None
        """
        for pre, _, node in RenderTree(self.root):
            if node.goal:
                print("{0}{1} {{{2}}}".format(pre, node.name, round(self.score, 2)))
            elif node.nt:
                print("{0}{1}".format(pre, node.name))
            else:
                print("{0}{1} [{2}]".format(pre, node.name, node.marker.name))
        print("\n")

    def dot_render(self):
        """
        Produces a graphviz PNG file of the tree.

        @return: None
        """
        UniqueDotExporter(self.root).to_picture("{0}.png".format(self.root.name))


class PlanLibrary:
    """
    A collection of Plans.
    """
    def __init__(self):
        self.plans = []
        self.explanations = []
        self.observations = []  # Temporal assumption: observation T happens after observation T-1
        self.eta = 0    # Normalization factor: all the scores will sum to 1
        self.initialize()

    def initialize(self):
        """
        Initializes the library.

        @return: None
        """
        # The plans are manually written and inserted in the library
        self.plans.extend([Plan(BreakfastNode()), Plan(LunchNode()), Plan(DrinkNode())])
        # Sets unique IDs by traversing the plans by breadth-first-search
        for plan in self.plans:
            id = 0
            for node in [node for node in PreOrderIter(plan.root)]:
                node.id = id
                id += 1
        # Sets the equivalency tables
        breakfast_equivalencies = [
            [(0, "food"), (1, "item")],
            [(1, "item"), (2, "food")],
            [(1, "destination"), (2, "vessel")],
            [(2, "vessel"), (3, "item")],
            [(3, "item"), (4, "item")],
            [(4, "item"), (5, "item")],
            [(4, "destination"), (5, "appliance")]
        ]
        drink_equivalencies = [
            [(0, "beverage"), (1, "item")],
            [(1, "item"), (2, "beverage")],
            [(1, "destination"), (2, "vessel")],
            [(2, "vessel"), (3, "item")],
            [(3, "item"), (4, "item")],
            [(4, "item"), (5, "item")],
            [(4, "destination"), (5, "appliance")]
        ]
        lunch_equivalencies = [
            [(0, "food"), (1, "food")],
            [(1, "food"), (2, "food")],
            [(2, "food"), (3, "item")],
            [(3, "item"), (4, "food")],
            [(3, "destination"), (4, "appliance")],
            [(2, "food"), (5, "item")],
            [(5, "food"), (6, "food")],
            [(5, "destination"), (6, "vessel")],
            [(6, "vessel"), (7, "item")],
            [(7, "item"), (8, "item")],
            [(8, "item"), (9, "item")],
            [(8, "destination"), (9, "appliance")]
        ]
        self.plans[0].equivalencies = breakfast_equivalencies
        self.plans[1].equivalencies = lunch_equivalencies
        self.plans[2].equivalencies = drink_equivalencies
        # The first explanations are the unobserved original trees
        #self.explanations.extend([Plan(BreakfastNode()), Plan(LunchNode()), Plan(DrinkNode())])
        self.explanations = copy.deepcopy(self.plans)

    def calculate_eta(self):
        """
        Calculates the normalization factor

        @return: None
        """
        try:
            self.eta = 1 / sum([e.score for e in self.explanations])
        except ZeroDivisionError:
            self.eta = 0

    def add_observation(self, observation, parameters):
        """
        Calculates new explanations for the new observation.

        @param observation: Name of a node
        @return: None
        """
        self.observations.append(observation)
        new_explanations = []   # We prepare a set of new explanations
        while self.explanations:
            explanation = self.explanations.pop()   # We pop because it will become obsolete
            indices = explanation.find_action_index(observation)    # Finds the relevant nodes
            if indices is not None:
                # The action is present in this explanation
                for index in indices:
                    # Process each possibility, generating a new tree for each one of them
                    if explanation.leaves[index].is_unobserved():
                        new_explanation: Plan = copy.deepcopy(explanation)
                        new_explanation.leaves[index].mark(Marker.OBSERVED)     # Marks in the new tree
                        # Sets the new parameters
                        for property, value in parameters.items():
                            new_explanation.leaves[index].parameters[property] = value
                        # Enforces the equivalencies todo
                        #new_explanation.enforce_equivalencies()
                        # All the unobserved nodes on the left-hand side should be marked as missed
                        for lhs_node in new_explanation.leaves[:index]:
                            if lhs_node.is_unobserved():
                                lhs_node.mark(Marker.MISSED)
                        new_explanation.calculate_score()   # Calculates the score of the individual explanation
                        new_explanations.append(new_explanation)
        self.explanations.extend(new_explanations)
        self.normalize_scores()

    def normalize_scores(self):
        """
        Normalizes the scores of all the observation in the [0, 1] range.

        @return: None
        """
        self.calculate_eta()
        for explanation in self.explanations:
            explanation.score *= self.eta

    def explain(self, sort=True):
        """
        Prints the current valid explanations, given the observations.

        @param sort: if True, it will show the more likely explanations first.
        @return: None
        """
        if len(self.explanations) == 0:
            print("No explanations!")
        else:
            if sort:
                set = sorted(self.explanations, key=lambda x: x.score, reverse=True)
            else:
                set = self.explanations
            for exp in set:
                exp.render()
