"""
Markovian finite-state machine for action prediction from movement primitives.
"""

from numpy.random import choice
from difflib import SequenceMatcher
import pandas as pd


class MarkovFSM:
    def __init__(self, chain, name, initial_probability):
        self.chain = chain
        self.name = name
        self.initial_prob = initial_probability

    def most_probable(self, state):
        """
        Returns the next most probable state.

        :param state: Initial state
        :return: State with highest transition probability.
        """
        row = self.chain.loc[state]
        return row.idxmax()

    def sample(self, size=3):
        """
        Generates a sample from the chain, based on the initial probability.

        :param size: number of states to generate
        :return: list of n=size elements
        """
        sample = []
        states = list(self.chain)
        current_state = choice(states, 1, p=self.initial_prob)[0]
        sample.append(current_state)
        for i in range(size-1):
            next_symbol = self.most_probable(current_state)
            sample.append(next_symbol)
            current_state = next_symbol
        return sample


class EnsembleFSM:
    def __init__(self, actions, action_names, field_names, initial_probabilities):
        # No safety checks implemented, beware!
        self.observations = []
        self.models = []
        # Constructs the models
        for i in range(len(actions)):
            chain = pd.DataFrame(actions[i], columns=field_names, index=field_names, dtype=float)
            model = MarkovFSM(chain, action_names[i], initial_probabilities[i])
            self.models.append(model)

    def add_observation(self, observation):
        """
        Adds an observation to the queue. Implements the Transition Analysis to filter out repetitions.

        :param observation: string, must be one of action_names
        :return: None
        """
        assert not isinstance(list, observation), "Observation must be a single item, not a list."
        states = list(self.models[0].chain)
        if observation in states:
            if len(self.observations) == 0 or self.observations[-1] != observation:
                self.observations.append(observation)
        else:
            print("[ERROR] Observation {0} does not match the ensemble state names!".format(observation))

    def empty_observations(self):
        """
        Resets the incremental observations.

        :return: None
        """
        self.observations = []

    def best_model(self):
        """
        Retrieves the best model which describes the given observations.

        :param observations: list of state sequences
        :return: Best fitting model, score, definite winner was found True/False
        """
        def similar(a, b):
            return SequenceMatcher(None, a, b).ratio()

        def refactor(list_sample):
            output = ''
            for word in list_sample:
                output += word[0]
            return output

        input_size = len(self.observations)
        top_score = -1.0
        top_model = None
        tie = False
        for model in self.models:
            sample = model.sample(size=input_size)
            score = similar(refactor(self.observations), refactor(sample))
            if score >= top_score:
                if score == top_score:  # There is a tie, register both names
                    if not isinstance(top_model, list):
                        top_model = [top_model]
                    top_model.append(model.name)
                    tie = True
                else:
                    top_score = score
                    top_model = model.name
        if not tie:
            return top_model, top_score, top_score >= 0.6
        else:
            return top_model, top_score / len(top_model), False     # A tie never wins


def get_trained_data():
    """
    A collection of the already trained data.

    :return: actions, action names, field names, initial probabilities
    """
    action1 = [
        [0, .033, .033, .9, .033],
        [.1, 0, 0, .9, 0],
        [.1, 0, 0, 0, .9],
        [.05, 0, .9, 0, .05],
        [.033, .033, 0, .9, .033]
    ]
    action2 = [
        [0, .05, .05, .45, .45],
        [.1, 0, 0, .9, 0],
        [.1, 0, 0, 0, .9],
        [.05, 0, .05, 0, .9],
        [.05, .05, 0, .9, 0]
    ]
    action3 = [
        [0, .9, .033, .033, .033],
        [.9, 0, 0, .1, 0],
        [.9, 0, 0, 0, .1],
        [.9, 0, .05, .0, .05],
        [.9, .05, 0, .05, 0]
    ]
    actions = [action1, action2, action3]
    field_names = ['STILL', 'WALK', 'TRANSPORT', 'PICK', 'PLACE']
    action_names = ["Pick and place", "Use", "Relocate"]

    t_prob = .95     # How much probable the most probable transition should be
    ip1 = [(1 - t_prob)/2, (1 - t_prob)/2, 0, t_prob, 0]
    ip2 = [(1 - t_prob)/3, (1 - t_prob)/3, (1 - t_prob)/3, t_prob, 0]
    ip3 = [t_prob, 0, (1 - t_prob)/3, (1 - t_prob)/3, (1 - t_prob)/3]
    initial_probabilities = [ip1, ip2, ip3]

    return actions, action_names, field_names, initial_probabilities


ensemble = EnsembleFSM(*get_trained_data())
