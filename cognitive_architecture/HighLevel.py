"""
High-Level cognitive architecture (plan recognition using a probabilistic context-free grammar)

"""

from cognitive_architecture.KnowledgeBase import kb
from cognitive_architecture.PlanLibrary import PlanLibrary
from datatypes.Prediction import Prediction
from util.exceptions import GoalNotRecognizedException
from multiprocessing import Process


class HighLevel(Process):
    def __init__(self, input_conn, output_conn):
        super().__init__()
        self.input_conn = input_conn
        self.output_conn = output_conn
        self.pl = PlanLibrary()
        # This dictionary contains pre-validated or denied sets of goal + target (e.g. LUNCH 'meal' is a valid goal)
        # They are stored here for persistence across multiple observations, which reduces resource-heavy calls
        self.history = {'valid': [], 'invalid': []}

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
        exps = self.pl.get_explanations()
        for exp in exps:
            # Goals with a None main parameter are skipped
            if not exp.is_parametrized():
                continue
            else:
                gs = exp.to_goal_statement()
                if gs in self.history['valid']:         # The goal was already approved
                    valid_explanations.append(exp)
                elif gs in self.history['invalid']:       # The goals was already disapproved
                    continue
                else:
                    if kb.verify_goal(gs):  # New goal, test it out. Note: this is a resource-heavy operation!
                        valid_explanations.append(exp)
                        self.history['valid'].append(gs)
                    else:
                        self.history['invalid'].append(gs)
        return valid_explanations

    @staticmethod
    def get_winner(explanations):
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

    def run(self) -> None:
        print("{0} process is running.".format(self.__class__.__name__))
        while True:
            # Waits for an observation to be available
            obs: Prediction = self.input_conn.get()
            self.add_observation(obs.name, obs.param)
            try:
                explanations = self.verify_explanations()
                goal = self.get_winner(explanations)
                if goal:
                    # If a goal was found, it communicates that to the CA
                    print("- - - - < GOAL: {0} > - - - -".format(goal))
                    self.output_conn.set(goal.to_prediction())
                    break   # HighLevel has completed its task
            except GoalNotRecognizedException:
                print("This robot has failed in recognizing the goal :(")
                break
