"""

This class uses an ontology to verify the correctness of the low-level predictions.

"""

from owlready2 import *
from util.PathProvider import path_provider


class ObservationStatement:
    def __init__(self, actor, action, target, destination):
        assert actor is not None and action is not None and target is not None and destination is not None, \
            "All the fields are mandatory and cannot be None."
        # For example: human PNP meal hobs
        self.actor = actor
        self.action = action
        self.target = target
        self.destination = destination
        # Corrects the "pick and place" action label, because it appears differently elsewhere:
        if self.action.lower() == "pick and place":
            self.action = "PNP"

    def __str__(self):
        return "{0} {1} {2} {3}".format(self.actor, self.action.upper(), self.target, self.destination)


class GoalStatement:
    def __init__(self, actor, goal, target):
        assert actor is not None and goal is not None and target is not None, \
            "All the fields are mandatory and cannot be None."
        # For example: human LUNCH meal
        self.actor = actor
        self.goal = goal
        self.target = target

    def __str__(self):
        return "{0} {1} {2}".format(self.actor, self.goal.upper(), self.target)


class KnowledgeBase:
    def __init__(self, ontology_name):
        ontofile = path_provider.get_ontology(ontology_name)
        self.onto = get_ontology(ontofile).load()
        self.op = [property.name for property in self.onto.object_properties()] # All the object properties names

    def find_individual(self, name):
        individuals = self.onto.individuals()
        for individual in individuals:
            if individual.name == name:
                return individual
        return None

    def verify(self, engine="pellet"):
        """
        Verifies the consistency of the ontology.

        @param engine: hermit or pellet
        @return: True if the ontology is consistent, False otherwise
        """
        assert engine in ["pellet", "hermit"], "Engine can be either 'hermit' or 'pellet'."
        try:
            if engine == "hermit":
                sync_reasoner_hermit()
            else:
                sync_reasoner_pellet()
            return True
        except OwlReadyInconsistentOntologyError:
            return False

    def verify_observation(self, observation_statement, debug=False):
        """
        Verify if an observation statement is consistent with the ontology.

        @param observation_statement: An ObservationStatement object representing the action of an actor in the world.
        @param debug: Verbose output
        @return: True or False
        """
        assert isinstance(observation_statement, ObservationStatement), "Input must be an ObservationStatement."
        # Finds the individuals referred by the statement
        actor = self.find_individual(observation_statement.actor)
        action = observation_statement.action
        target = self.find_individual(observation_statement.target)
        destination = self.find_individual(observation_statement.destination)
        if debug:
            print("{0} {1} {2} {3}".format(actor, action, target, destination))
        # Analyse the action
        p = action.lower()  # The name of the action is the name of the object property, es: cook_target
        pt = p + '_target'
        pd = p + '_destination'
        # Verify the existence of the properties for that action
        if pt in self.op and pd in self.op:
            # Set the attributes
            setattr(actor, pt, target)
            setattr(actor, pd, destination)
            # Verification
            result = self.verify()
            # Resets the individuals for the next tests
            setattr(actor, pt, None)
            setattr(actor, pd, None)
        else:
            if debug:
                print("Action not found")
            result = False    # The action was not recognized
        if debug:
            print("Is the action consistent in the ontology? {0}".format(result))
        return result

    def verify_goal(self, goal_statement, debug=False):
        """
        Verify if a goal statement is consistent with the ontology.

        @param goal_statement: A GoalStatement object representing the goal of an actor in the world.
        @param debug: Verbose output
        @return: True or False
        """
        assert isinstance(goal_statement, GoalStatement), "Input must be a GoalStatement."
        # Finds the individuals referred by the statement
        actor = self.find_individual(goal_statement.actor)
        goal = self.find_individual(goal_statement.goal.lower())
        target = self.find_individual(goal_statement.target)
        if debug:
            print("{0} {1} {2}".format(actor, goal, target))
        # Set the attributes
        setattr(actor, "has_goal", goal)
        setattr(goal, "involves_object", target)
        # Verification
        result = self.verify()
        # Resets the individuals for the next tests
        setattr(actor, "has_goal", None)
        setattr(goal, "involves_object", None)
        if debug:
            print("Is the goal consistent in the ontology? {0}".format(result))
        return result


kb = KnowledgeBase('kitchen_onto')