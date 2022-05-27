"""

These classes model the Plan Library to perform probabilistic goal recognition.

"""

import copy
from anytree import NodeMixin, RenderTree, PreOrderIter
from anytree.exporter import UniqueDotExporter
from anytree.search import findall
from enum import Enum
from cognitive_architecture.KnowledgeBase import ObservationStatement, GoalStatement
from datatypes.Prediction import Prediction
from util.exceptions import GoalNotRecognizedException


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

    def to_observation_statement(self, actor):
        """
        Returns an ObservationStatement.

        @param actor: Name of the actor (could be used for the human or the robot)
        @return: ObservationStatement
        """
        parameters = list(self.parameters.values())
        return ObservationStatement(actor, self.name, parameters[0], parameters[1])

    def __str__(self):
        return "{0} {1}".format(self.name, self.parameters)


# Terminal nodes

class PnPNode(PLNode):
    def __init__(self, id=0, parent=None, children=None):
        super().__init__(action="PICK AND PLACE", id=id, parent=parent, children=children)
        self.nt = False
        self.parameters = {
            "item": None,
            "destination": None
        }


class EatNode(PLNode):
    def __init__(self, id=0, parent=None, children=None):
        super().__init__(action="EAT", id=id, parent=parent, children=children)
        self.nt = False
        self.parameters = {
            "food": None,
            "vessel": "plate"
        }


class SipNode(PLNode):
    def __init__(self, id=0, parent=None, children=None):
        super().__init__(action="SIP", id=id, parent=parent, children=children)
        self.nt = False
        self.parameters = {
            "beverage": None,
            "vessel": "glass"
        }


class WashNode(PLNode):
    def __init__(self, id=0, parent=None, children=None):
        super().__init__(action="WASH", id=id, parent=parent, children=children)
        self.nt = False
        self.parameters = {
            "item": None,
            "appliance": "sink"
        }


class CookNode(PLNode):
    def __init__(self, id=0, parent=None, children=None):
        super().__init__(action="COOK", id=id, parent=parent, children=children)
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
            "food": None,
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
            "vessel": "plate",
            "wash": "sink"
        }
        self.children = [PnPNode(), EatNode(), CleanNode()]


class LunchNode(PLNode):
    def __init__(self, id=0, parent=None, children=None):
        super().__init__(action="Lunch", id=id, parent=parent, children=children)
        self.goal = True
        self.parameters = {
            "food": None,
            "appliance": "hobs",
            "vessel": "plate",
            "wash": "sink"
        }
        self.children = [PrepareMealNode(), EatNode(), CleanNode()]


class DrinkNode(PLNode):
    def __init__(self, id=0, parent=None, children=None):
        super().__init__(action="Drink", id=id, parent=parent, children=children)
        self.goal = True
        self.parameters = {
            "beverage": None,
            "vessel": "glass",
            "wash": "sink"
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

    def get_name_and_parameters(self):
        """
        Returns the name of the overall plan (aka the name of the root node)

        @return: string name and dict parameters of the root node
        """
        return self.root.name, self.root.parameters

    def get_first_parameter(self, as_dict=False):
        """
        Returns the first item of the parameters dictionary, which should be the main one.

        @param as_dict: If True, returns a dictionary
        @return: Either the first parameters value or the key:value dictionary
        """
        if as_dict:
            d = {}
            (k, v), *rest = self.root.parameters.items()
            d[k] = v
            return d
        else:
            return list(self.root.parameters.values())[0]

    def is_parametrized(self):
        """
        Is the main parameter defined or does it have a None value?
        This is useful to know if a goal was correctly parametrized or if the observation(s) are not sufficient to
        determine the target of the intention.

        @return: True or False
        """
        return True if self.get_first_parameter() is not None else False

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
        nodelist = findall(self.root, filter_=lambda node: node.id == id, maxcount=1)
        if nodelist:
            return nodelist[0]
        return None

    def calculate_score(self):
        """
        Calculates the score of the tree as: % of observed nodes * (1 - % of missed nodes).
        This score follows Occam's Razor principle, favoring the simplest explanation and penalizing explanations that
        contain too many missed nodes.

        @return: None
        """
        frontier = self.root.leaves
        completed = 0   # % of observed nodes
        missed = 0      # % of missed nodes
        for leaf in frontier:
            if leaf.is_observed():
                completed += 1
            elif leaf.is_missed():
                missed += 1
        completed = completed / len(frontier)
        missed = missed / len(frontier)
        self.score = completed * (1 - missed)

    def propagate_equivalencies(self, observation):
        """
        Propagates the equivalencies using the appropriate table. First from observation to root, then from root to all
        the other nodes.

        @return: None
        """
        def process_equivalency(couple, debug=False):
            """
            Processes a single equivalency couple.

            @param couple: [(nodeA_id, paramA), (nodeB_id, paramB)]
            @param debug: if True, activate verbose output
            @return: None
            """
            nodeA_id = couple[0][0]
            paramA = couple[0][1]
            nodeB_id = couple[1][0]
            paramB = couple[1][1]
            nodeA = self.get_node_by_id(nodeA_id)
            nodeB = self.get_node_by_id(nodeB_id)
            if debug:
                print("\nMatching:")
                print("Node {0} (id {1}), parameter {2} = {3}".format(nodeA.name, nodeA_id, paramA,
                                                                      nodeA.parameters[paramA]))
                print("Node {0} (id {1}), parameter {2} = {3}".format(nodeB.name, nodeB_id, paramB,
                                                                      nodeB.parameters[paramB]))
            # If the two parameter values are different, fill in the None values, if any
            if nodeA.parameters[paramA] != nodeB.parameters[paramB]:
                if nodeA.parameters[paramA] is None:
                    nodeA.parameters[paramA] = nodeB.parameters[paramB]
                elif nodeB.parameters[paramB] is None:
                    nodeB.parameters[paramB] = nodeA.parameters[paramA]
                # Should something happen when a conflict is detected?
                #else:
                    #print("There is a conflict!\nNode {0}, parameter {1} = {2}\nNode {3}, parameter {4} = {5}".format(
                    #    nodeA.name, paramA, nodeA.parameters[paramA], nodeB.name, paramB, nodeB.parameters[paramB]
                    #))

        # First, only propagate the equivalencies related to the observation
        for couple in [couple for couple in self.equivalencies if couple[1][0] == observation]:
            process_equivalency(couple)
        # Then, propagate every other equivalency
        for couple in self.equivalencies:
            if couple[0][1] != observation:
                process_equivalency(couple)

    def get_frontier(self):
        """
        Obtains a list of the unobserved actions.

        @return: list of unobserved Nodes
        """
        return [node for node in self.leaves if node.marker == Marker.UNOBSERVED]

    def to_goal_statement(self):
        """
        Creates a GoalStatement based on this plan, used for validation in the KnowledgeBase.

        @return: GoalStatement
        """
        # todo this assumes the existence of only one human, has to be changed in the future
        return GoalStatement("human", self.root.name, list(self.root.parameters.values())[0], self.get_frontier())

    def to_prediction(self):
        """
        Converts a Plan into a Prediction object.

        @return: Prediction
        """
        return Prediction(self.root.name, self.get_first_parameter(as_dict=True), self.score, self.get_frontier())

    def render(self):
        """
        Render the tree in the terminal.

        @return: None
        """
        for pre, _, node in RenderTree(self.root):
            if node.goal:
                print("[{0}] {1}{2}".format(round(self.score, 2), pre, node))
            elif node.nt:
                print("{0}{1}".format(pre, node))
            else:
                print("{0}{1} [{2}]".format(pre, node, node.marker.name))
        print("\n")

    def dot_render(self):
        """
        Produces a graphviz PNG file of the tree.

        @return: None
        """
        UniqueDotExporter(self.root).to_picture("{0}.png".format(self.root.name))

    def __str__(self):
        return "{0} with parameters {1}".format(self.root.name, self.get_first_parameter(as_dict=True))


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
            for id, node in enumerate([node for node in PreOrderIter(plan.root)]):
                node.id = id
        # Sets the equivalency tables
        # TUTORIAL: How to write the equivalencies?
        # First element should be the ROOT node, second item should be the child
        breakfast_equivalencies = [
            [(0, "food"), (1, "item")],
            [(0, "vessel"), (1, "destination")],
            [(0, "food"), (2, "food")],
            [(0, "vessel"), (2, "vessel")],
            [(0, "vessel"), (3, "item")],
            [(0, "vessel"), (4, "item")],
            [(0, "wash"), (4, "destination")],
            [(0, "vessel"), (5, "item")],
            [(0, "wash"), (5, "appliance")],
        ]
        lunch_equivalencies = [
            [(0, "food"), (1, "food")],
            [(0, "food"), (2, "food")],
            [(0, "food"), (3, "item")],
            [(0, "appliance"), (3, "destination")],
            [(0, "food"), (4, "food")],
            [(0, "food"), (5, "item")],
            [(0, "vessel"), (5, "destination")],
            [(0, "food"), (6, "food")],
            [(0, "vessel"), (7, "item")],
            [(0, "vessel"), (8, "item")],
            [(0, "wash"), (8, "destination")],
            [(0, "vessel"), (9, "item")],
            [(0, "vessel"), (9, "appliance")],
        ]
        drink_equivalencies = [
            [(0, "beverage"), (1, "item")],
            [(0, "vessel"), (1, "destination")],
            [(0, "beverage"), (2, "beverage")],
            [(0, "vessel"), (2, "vessel")],
            [(0, "vessel"), (3, "item")],
            [(0, "vessel"), (4, "item")],
            [(0, "wash"), (4, "destination")],
            [(0, "vessel"), (5, "item")],
            [(0, "wash"), (5, "appliance")],
        ]
        self.plans[0].equivalencies = breakfast_equivalencies
        self.plans[1].equivalencies = lunch_equivalencies
        self.plans[2].equivalencies = drink_equivalencies
        # The first explanations are the unobserved original trees
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

    def add_observation(self, observation, parameters, ignore=False):
        """
        Calculates new explanations for the new observation.

        @param observation: Name of a node
        @param parameters: dict of parameters
        @param ignore: if True, ignores observations which would refute old observations
        @return: None
        """
        self.observations.append(observation)
        new_explanations = []   # We prepare a set of new explanations
        while self.explanations:
            # We pop the old explanation because the new observation will either refute or refine it
            explanation = self.explanations.pop()
            indices = explanation.find_action_index(observation)    # Finds the relevant nodes
            if indices is not None:
                # The action is present in this explanation
                for index in indices:
                    # Process each possibility, generating a new tree for each one of them
                    if explanation.leaves[index].is_unobserved():
                        new_explanation: Plan = copy.deepcopy(explanation)
                        node_of_interest = new_explanation.leaves[index]
                        node_of_interest.mark(Marker.OBSERVED)     # Marks in the new tree
                        # Matches the parameter names
                        matched_params = dict(zip(node_of_interest.parameters.keys(), parameters.values()))
                        # Sets the new parameters
                        node_of_interest.parameters = matched_params
                        # Enforces the equivalencies
                        new_explanation.propagate_equivalencies(node_of_interest.id)
                        # All the unobserved nodes on the left-hand side should be marked as missed
                        for lhs_node in new_explanation.leaves[:index]:
                            if lhs_node.is_unobserved():
                                lhs_node.mark(Marker.MISSED)
                        new_explanation.calculate_score()   # Calculates the score of the individual explanation
                        new_explanations.append(new_explanation)
                    elif ignore:
                        # If in 'ignore' mode, re-insert an explanation for which no observations could be matched
                        new_explanations.append(explanation)
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

    def get_explanations(self, render=False):
        """
        Prints the current valid explanations, given the observations.

        @return: list of Plans, ordered by probability
        """
        output = None
        if len(self.explanations) == 0:
            print("No explanations!")
            raise GoalNotRecognizedException
        else:
            output = sorted(self.explanations, key=lambda x: x.score, reverse=True)
            if render:
                for exp in output:
                    exp.render()
        return output
