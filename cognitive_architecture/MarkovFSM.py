"""
Markovian finite-state machine for action prediction from movement primitives.
"""

from numpy.random import choice
from difflib import SequenceMatcher
import pandas as pd
from statistics import mode
import numpy as np


class MarkovFSM:
    def __init__(self, chain, name, initial_probability):
        self.chain = chain
        self.name = name
        self.initial_prob = initial_probability
        self.sample = []

    def most_probable(self, state):
        """
        Returns the next most probable state.

        :param state: Initial state
        :return: State with the highest transition probability.
        """
        row = self.chain.loc[state]
        return row.idxmax()

    def sampling(self, start=None):
        """
        Generates a sample from the chain. If a starting node is not explicitly provided, it will be based on the
        initial probability distribution.

        :param start: optional, forces the first state of the sampling
        :return: None
        """
        self.sample = []     # Resets the sample
        states = list(self.chain)
        if start:
            if start in states:
                current_state = start
            else:
                print("[ERROR] Sample {0} not in states {1}. Ignoring.".format(start, states))
                current_state = choice(states, 1, p=self.initial_prob)[0]
        else:
            current_state = choice(states, 1, p=self.initial_prob)[0]
        self.sample.append(current_state)
        while len(self.sample) < 6:
            next_symbol = self.most_probable(current_state)
            self.sample.append(next_symbol)
            current_state = next_symbol


class EnsembleFSM:
    def __init__(self, actions, action_names, field_names, initial_probabilities, debug=True):
        # No safety checks implemented, beware!
        self.observations = []      # Raw list with all the observations
        self.filtered_observations = []     # Observations filtered by transition and sliding window
        self.models = []
        self.debug = debug
        # Constructs the models
        for i in range(len(actions)):
            chain = pd.DataFrame(actions[i], columns=field_names, index=field_names, dtype=float)
            model = MarkovFSM(chain, action_names[i], initial_probabilities[i])
            self.models.append(model)

    def init_sampling(self, start=None):
        """
        Performs a sampling on each model, eventually providing a starting point.

        @param start: optional, forces the first state of the sampling
        @return: None
        """
        for model in self.models:
            model.sampling(start)

    def is_first_insertion(self):
        """
        Returns True if thee filtered observation window has only one element.
        This is for sampling initialization purposes.

        @return: True of False
        """
        return True if len(self.filtered_observations) == 0 else False

    def add_observation(self, observation, w=3):
        """
        Adds an observation to the queue. Implements the Transition Analysis to filter out repetitions.

        :param observation: string, must be one of action_names
        :param w: sliding window size
        :param debug: if True, enables verbose debug output
        :return: None
        """
        assert not isinstance(observation, list), "Observation must be a single item, not a list."
        assert isinstance(w, int) and w > 0, "The sliding window size w must be a positive integer."
        states = list(self.models[0].chain)
        if observation not in states:
            print("[ERROR] Observation '{0}' does not match the ensemble state names!".format(observation))
        else:
            self.observations.append(observation)
            #if self.debug:
            #    print("[DEBUG] Raw observations: {0}".format(self.observations))
            # If at least W observations are present, apply the sliding window
            if len(self.observations) >= w:
                window = self.observations[-w:]
                filtered_observation = mode(window)     # central tendency
                # print("Window: {0}\nFiltered: {1}".format(window, filtered_observation))
                # Transition filtering
                if len(self.filtered_observations) == 0 or self.filtered_observations[-1] != filtered_observation:
                    self.filtered_observations.append(filtered_observation)
            if self.debug:
                print("[DEBUG] Filtered observations: {0}".format(self.filtered_observations))

    def empty_observations(self):
        """
        Resets the incremental observations.

        :return: None
        """
        self.observations = []
        self.filtered_observations = []

    def best_model(self, threshold=0.7):
        """
        Retrieves the best model which describes the given observations.

        :param threshold: confidence over which to declare a model as a winner
        :return: Best fitting model, score, definite winner was found True/False
        """
        def similar(a, b):
            return SequenceMatcher(None, a, b).ratio()

        def refactor(list_sample):
            output = ''
            for word in list_sample:
                output += word[0]
            return output

        input_size = len(self.filtered_observations)
        if input_size == 1:
            # If it's the very first insertion, initiate the sample with the detected movement
            self.init_sampling(self.filtered_observations[0])
        top_score = -1.0
        top_model = None
        tie = False
        for model in self.models:
            sample = model.sample
            score = similar(refactor(self.filtered_observations), refactor(sample))
            if score >= top_score:
                if score == top_score:  # There is a tie, register both names
                    if not isinstance(top_model, list):
                        top_model = [top_model]
                    top_model.append(model.name)
                    tie = True
                else:
                    top_score = score
                    top_model = model.name
                    tie = False
        if not tie:
            return top_model, top_score, top_score >= threshold
        else:
            return top_model, top_score / len(top_model), False     # A tie never wins

    def get_scores(self):
        """
        Calculates the normalized scores that each model obtains on the observations.

        @return: Array of scores.
        """
        def similar(a, b):
            return SequenceMatcher(None, a, b).ratio()

        def refactor(list_sample):
            output = ''
            for word in list_sample:
                output += word[0]
            return output

        input_size = len(self.filtered_observations)
        if input_size == 1:
            # If it's the very first insertion, initiate the sample with the detected movement
            self.init_sampling(self.filtered_observations[0])
        scores = [0, 0, 0]
        for i, model in enumerate(self.models):
            sample = model.sample
            score = similar(refactor(self.filtered_observations), refactor(sample))
            scores[i] = score
        # Normalization
        #eta = 1 / sum(scores)
        #for i, score in enumerate(scores):
        #    scores[i] *= eta
        return scores

    def evaluate(self):
        """
        Debug function to evaluate the training data. Prints some accuracy metrics.
        Does not work anymore due to modifications on the sampling algorithm.

        :return: None
        """
        lost = []
        for j in range(100):
            losers = 0
            for i in range(1000):
                self.filtered_observations = ['PICK', 'PLACE', 'PICK']
                model, score, winner = self.best_model()
                if not winner:
                    losers += 1
            print("Total failed: {0}/{1} ({2}%)".format(losers, 1000, losers / 1000.0))
            lost.append(losers / 1000.0)
        lost = np.array(lost)
        print("\nMean: {0}\nVariance:{1}".format(lost.mean(), lost.std()))


class ForgivenessWindow:
    """
    Models a forgiveness window, which ignores instances of ObservationY if ObservationX was sensed last, for a limited
    time frame.
    """
    def __init__(self, rival_pairs, n=1):
        """
        Constructor.

        @param rival_pairs: list of tuples, [(ObsX, ObsY), ...]. ObsY will be banned if occurring within n timestamps
        after ObsX
        @param n: width of the forgiveness window, i.e. how long to ignore ObsY for
        """
        # Sanity check
        assert isinstance(rival_pairs, list) and all([isinstance(x, tuple) for x in rival_pairs]), \
            "rival_pairs should be a list of tuples"
        self.rival_pairs = rival_pairs
        self.n = n
        self.previous_obs_name = None
        self.previous_obs_time = 0

    def insert(self, observation, time):
        """
        Inserts a new observation and verifies if it will be accepted or not

        @param observation: Name of the observation, str
        @param time: Timestamp of the observation, int
        @return: True if accepted, else False
        """
        assert isinstance(observation, str), "Observation must be a str value"
        assert isinstance(time, int), "Time must be an int value"
        accepted = True
        for obsx, obsy in self.rival_pairs:
            if obsx == self.previous_obs_name and obsy == observation:
                accepted = True if (time - self.previous_obs_time > self.n) else False
                break
        self.previous_obs_name = observation
        self.previous_obs_time = time
        return accepted



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

    t_prob = .99     # How much probable the most probable transition should be
    ip1 = [(1 - t_prob)/2, (1 - t_prob)/2, 0, t_prob, 0]
    ip2 = [(1 - t_prob)/3, (1 - t_prob)/3, (1 - t_prob)/3, t_prob, 0]
    ip3 = [t_prob, 0, (1 - t_prob)/3, (1 - t_prob)/3, (1 - t_prob)/3]
    initial_probabilities = [ip1, ip2, ip3]

    return actions, action_names, field_names, initial_probabilities


ensemble = EnsembleFSM(*get_trained_data(), debug=False)
