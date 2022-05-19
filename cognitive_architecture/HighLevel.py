"""
High-Level cognitive architecture (plan recognition using a probabilistic context-free grammar)

"""

from cognitive_architecture.KnowledgeBase import kb
from cognitive_architecture.PlanLibrary import PlanLibrary
from util.exceptions import GoalNotRecognizedException


class HighLevel:
    def __init__(self):
        self.pl = PlanLibrary()

    def add_observation(self, observation, parameters):
        """
        Wrapper to add an observation to the PlanLibrary.

        @param observation: name of the action observed, str
        @param parameters: parameters of the action, dict
        @return: None
        """
        assert isinstance(observation, str), "Observation must be the name of an action."
        assert isinstance(parameters, dict), "Parameters must be a dictionary of parameters."
        self.pl.add_observation(observation, parameters)

    def verify_explanations(self):
        """
        Verifies the explanation's validity using the knowledge base

        @return: A list of valid Plan
        """
        valid_explanations = []     # The final output
        # Here we record goals which we have proven to be false, e.g. any variation of Breakfast with target = 'meal'
        # We do this to minimize the number of verifications that we perform, since they are resource-demanding
        blacklist = []
        whitelist = []
        exps = self.pl.get_explanations()
        for exp in exps:
            # Goals with a None main parameter are skipped
            if not exp.is_parametrized():
                continue
            else:
                gs = exp.to_goal_statement()
                if gs in whitelist:         # The goal was already approved
                    valid_explanations.append(exp)
                elif gs in blacklist:       # The goals was already disapproved
                    continue
                else:                       # New goal, test it out
                    if kb.verify_goal(gs):
                        valid_explanations.append(exp)
                        whitelist.append(gs)
                    else:
                        blacklist.append(gs)
        return valid_explanations

    def get_winner(self, explanations):
        """
        Retrieve a winner explanation, if it exists.

        @param explanations: list of plans (possibly pre-validated)
        @return: One single Plan, or None if no or multiple possibilities exist
        """
        if len(explanations) == 0:
            # No explanation found
            raise GoalNotRecognizedException
        elif len(explanations) == 1:
            # One explanation found
            return explanations[0]
        elif explanations[0].score != explanations[1].score:
            # Multiple explanations, but with a clear winner
            return explanations[0]
        else:
            # Multiple explanations with some ties between the top scored
            return None

    def process(self, observation, parameters):
        """
        Main point of contact with this class. Adds the observation, creates explanations, filters out the invalid ones
        and selects a winner.

        @param observation: name of the action observed, str
        @param parameters: parameters of the action, dict
        @return: A Plan, if a winner exists, or None
        """
        self.add_observation(observation, parameters)
        explanations = self.verify_explanations()
        return self.get_winner(explanations)
