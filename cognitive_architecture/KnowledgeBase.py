"""

This class uses an ontology to verify the correctness of the low-level predictions.

"""

from owlready2 import *
from util.PathProvider import path_provider


class ObservationStatement:
    def __init__(self, actor, action, target, destination):
        # For example: human PNP meal hobs
        self.actor = actor
        self.action = action
        self.target = target
        self.destination = destination
        self.valid = None
        # Corrects the "pick and place" action label, because it appears differently elsewhere:
        if self.action.lower() == "pick and place":
            self.action = "PNP"

    @staticmethod
    def from_NT(nt):
        # Creates a statement from the frontier of an explanation
        action = nt._ch.removesuffix('NT')
        target = nt._params[0][1]
        try:
            destination = nt._params[1][1]
        except IndexError:
            destination = None
        return ObservationStatement("human", action, target, destination)

    def __str__(self):
        return "{0} {1} {2} {3}".format(self.actor, self.action.upper(), self.target, self.destination)


class GoalStatement:
    def __init__(self, actor, goal, target, frontier=None):
        assert actor is not None and goal is not None and target is not None, \
            "All the fields are mandatory and cannot be None."
        # For example: human LUNCH meal
        self.actor = actor
        self.goal = goal
        self.target = target
        self.frontier = frontier
        self.valid = None

    def __str__(self):
        return "{0} {1} {2}\nFrontier: {3}".format(self.actor, self.goal.upper(), self.target,
                                                   [str(f) for f in self.frontier])


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

    def verify(self, engine="pellet", infer=False):
        """
        Verifies the consistency of the ontology.

        @param engine: hermit or pellet
        @param infer: if True, allows inference for property values through SWRL rules
        @return: True if the ontology is consistent, False otherwise
        """
        assert engine in ["pellet", "hermit"], "Engine can be either 'hermit' or 'pellet'."
        print("[KB] Reasoning...")
        try:
            if engine == "hermit":
                sync_reasoner_hermit(infer_property_values=infer)
            else:
                sync_reasoner_pellet(infer_property_values=infer)
            return True
        except OwlReadyInconsistentOntologyError:
            return False

    def verify_observation(self, observation_statement, infer=False, debug=False):
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
            if destination is not None:
                # Destination might be none if the observation comes from the frontier
                setattr(actor, pd, destination)
            # Verification
            observation_statement.valid = self.verify(engine="pellet", infer=infer)
            # If we are inferring properties, these have to be saved in the original statement
            if infer:
                target_individual = getattr(actor, pt)
                observation_statement.target = target_individual.name if target_individual is not None else None
                dest_individual = getattr(actor, pd)
                observation_statement.destination = dest_individual.name if dest_individual is not None else None
            # Resets the individuals for the next tests
            setattr(actor, pt, None)
            setattr(actor, pd, None)
        else:
            if debug:
                print("Action not found")
            result = False    # The action was not recognized
        if debug:
            print("Is the action consistent in the ontology? {0}".format(observation_statement.valid))
        return observation_statement.valid

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
        goal_statement.valid = self.verify()
        # Resets the individuals for the next tests
        setattr(actor, "has_goal", None)
        setattr(goal, "involves_object", None)
        if debug:
            print("Is the goal consistent in the ontology? {0}".format(goal_statement.valid))
        return goal_statement.valid

    def infer_frontier(self, goal):
        """
        This function allows the reasoner to fill in some details that might be missing from the CRADLE explanation.

        @param goal: GoalStatement which contains the frontier to be analyzed
        @return: None
        """
        assert isinstance(goal, GoalStatement), "goal has to be an instance of GoalStatement."
        for f in goal.frontier:
            self.verify_observation(f, infer=True)




kb = KnowledgeBase('kitchen_onto')