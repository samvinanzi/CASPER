"""
High-Level cognitive architecture (plan recognition using a probabilistic context-free grammar)

"""

from cognitive_architecture.KnowledgeBase import kb
from cognitive_architecture.PlanLibrary import PlanLibrary, Plan
from datatypes.Prediction import Prediction
from util.exceptions import GoalNotRecognizedException
from multiprocessing import Process
from itertools import groupby
from util.Logger import Logger
import time


class HighLevel(Process):
    def __init__(self, input_conn, output_conn, verification):
        super().__init__()
        self.input_conn = input_conn
        self.output_conn = output_conn
        self.verification = verification    # Enable / disable formal verification
        self.pl = PlanLibrary()
        self.logger = Logger()
        # This dictionary contains pre-validated or denied sets of goal + target (e.g. LUNCH 'meal' is a valid goal)
        # They are stored here for persistence across multiple observations, which reduces resource-heavy calls
        self.history = {'valid': [], 'invalid': []}

    def add_observation(self, observation, parameters, ignore=False):
        """
        Wrapper to add an observation to the PlanLibrary.

        @param observation: name of the action observed, str
        @param parameters: parameters of the action, dict
        @return: None
        """
        assert isinstance(observation, str), "Observation must be the name of an action."
        assert isinstance(parameters, dict), "Parameters must be a dictionary of parameters."
        self.pl.add_observation(observation, parameters, ignore)

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
                # Note: I'm not using KnowledgeBase's intrinsic history because there is a risk of conflict with
                # LowLevel and it's not worth implementing additional synchronization just for this.
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

    @staticmethod
    def make_plan(goal: Plan):
        """
        Creates a collaborative plan based on a goal's frontier. It calculates which actions are executable by the robot
        and finds the longest subsequence of actions that the agent can perform.

        @param goal: Plan
        @return: list of PLNodes
        """
        # Ignore the very first action: it's likely the human will perform that one
        original_frontier = goal.get_frontier()
        new_frontier = original_frontier[1:]
        # Verifies which actions are performable by the robot
        performable = [False] * len(new_frontier)
        for i, action in enumerate(new_frontier):
            obs = action.to_observation_statement("tiago")  # todo multiple robots
            valid = kb.verify_observation(obs)
            performable[i] = valid
        # Finds the longest subsequence
        start = 0
        runs = []
        for key, run in groupby(performable):
            length = sum(1 for _ in run)
            runs.append((start, start + length - 1))
            start += length
        result = max(runs, key=lambda x: x[1] - x[0])
        # todo This works on the assumption that the human does not operate after the robot.
        human_side = new_frontier[:result[0]]
        robot_side = new_frontier[result[0]:result[1]+1]
        return human_side, robot_side

    def has_human_completed(self, human_plan):
        """
        Verifies if a human plan was completed. This happens when all the nodes are either observed or missed.
        @param human_plan: list of PLNodes
        @return: True or False
        """
        # Each observation can produce multiple explanations. It is important to verify all of them.
        for explanation in self.pl.explanations:
            for action in human_plan:
                node_in_pl = explanation.get_node_by_id(action.id)
                if node_in_pl.is_unobserved():
                    return False
            return True

    def run(self) -> None:
        print("{0} process is running.".format(self.__class__.__name__))
        human_plan = None
        robot_plan = None
        t = time.time()
        while True:
            # Are we in phase 2 (plan filling)?
            phase1 = True if human_plan is None and robot_plan is None else False
            # If the human plan is empty, the robot should not wait to act
            if phase1 or len(human_plan) > 0:
                # Waits for an observation to be available
                obs: Prediction = self.input_conn.get()
                # Adds it to the plan library
                self.add_observation(obs.name, obs.param, ignore=not phase1)
            if phase1:
                # PHASE 1: if HL is still searching for a goal, produce explanations
                try:
                    explanations = self.verify_explanations() if self.verification else self.pl.get_explanations()
                    goal = self.get_winner(explanations)
                    if goal:
                        t = time.time() - t
                        print("TIME: {0}".format(t))
                        print("- - - - < GOAL: {0} > - - - -".format(goal))
                        # Cancel every other explanation
                        self.pl.explanations = [goal]
                        # Decide on a collaboration plan
                        human_plan, robot_plan = self.make_plan(goal)
                        print("Waiting for the human to accomplish the following:")
                        for action in human_plan:
                            print("\t{0}{1}".format(action.name, action.parameters))
                except GoalNotRecognizedException:
                    print("This robot has failed in recognizing the goal :(")
                    break
            else:
                # PHASE 2: HL has already predicted a goal and is now waiting for the right time to act
                # Log the data
                log_data = {
                    'observed': self.pl.explanations[0].get_number_observed(),
                    'missed': self.pl.explanations[0].get_number_missed(),
                    'waiting': len(human_plan),
                    'planned': len(robot_plan),
                    'goal': goal,
                    'time': round(t, 2)
                }
                self.logger.log(log_data)
                #return # todo enable or multiple entries will be recorded
                if self.has_human_completed(human_plan):
                    # The human has completed their part of the plan. Signal the CA.
                    self.output_conn.set(robot_plan)
                    break
