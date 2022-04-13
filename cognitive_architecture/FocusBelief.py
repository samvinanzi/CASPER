"""
Models the robot's belief on the human's focus.
"""

from cognitive_architecture.Episode import ObjectFrame
import matplotlib.pyplot as plt
import threading
from util.PathProvider import path_provider
import csv
from queue import Queue
from collections import Counter


class Window:
    """
    This class models a sliding window.
    """
    def __init__(self, size=4, score=3):
        assert score < size, "Score must be less than the window size."
        self.items = Queue(maxsize=size)     # FIFO queue
        self.score = score
        self.most_recent_winner = None

    def add(self, item):
        """
        Adds an item to the window

        @param item: item to insert
        """
        if self.items.full():
            self.items.get()
        self.items.put(item)

    def get_items(self):
        """
        Returns the elements inside the window.

        @return: list of elements
        """
        return list(self.items.queue)

    def check_winner(self):
        """
        Checks if one of the items is recurring at least Score times.

        @return: True or False
        """
        if self.items.empty():
            return False
        c = Counter(self.get_items())
        item, frequency = c.most_common(1)[0]
        if frequency >= self.score:
            self.most_recent_winner = item
            return True
        else:
            return False

    def contains_winner(self, item_list):
        """
        Checks if one of the items supplied as a parameters is the one that has already been designed as a winner.

        @param item_list: List of item names
        @return: Winner (if present) and list of remaining items
        """
        if self.most_recent_winner in item_list:
            return self.most_recent_winner, [item for item in item_list if item != self.most_recent_winner]
        else:
            return None, item_list


class FocusBelief:
    class FocusItem:
        def __init__(self, name, p):
            self.name = name
            self.p = p

    def __init__(self, human_name, w_qdc=.8, w_qtc=.2, threshold=.5, epsilon=.05):
        self.name = human_name
        self.w_qdc = w_qdc
        self.w_qtc = w_qtc
        self.threshold = threshold
        self.epsilon = epsilon
        self.raw_values = {}     # Non-normalized in [0, 1]
        self.normalized_probabilities = {}
        # Probability mapping
        self.qdc_map = {
            'FAR': 0,
            'MEDIUM': .125,
            'NEAR': .25,
            'TOUCH': .5
        }
        self.qtc_map = {
            '+': 0,
            '-': .25,
            '0': .5
        }
        self.target = None
        self.destination = None
        self.target_window = Window()
        self.destination_window = Window(size=5, score=3)
        self.log = path_provider.get_csv('focus_belief.csv')
        # Empties the contents of the log file
        with open(self.log, 'w'):
            pass

    def get_object_names(self):
        """
        Returns the names of the objects already in memory.

        :return: list of names
        """
        return list(self.raw_values.keys())

    def p(self, object):
        """
        Calculates the probability for a given set of object QSRs.

        :param object: ObjectFrame of interest.
        :return: probability in [0.0, 1.0]
        """
        assert isinstance(object, ObjectFrame), "Parameter must be of type ObjectFrame"
        return (self.w_qdc * self.qdc_map[object.QDC] + self.w_qtc * self.qtc_map[object.QTC]) / (1 + object.theta)

    def normalize_all(self):
        """
        Normalizes the raw values into probabilities.

        :return: None
        """
        if self.raw_values:
            total = sum(self.raw_values.values())
            if total == 0.0:
                return
            for key, value in self.raw_values.items():
                self.normalized_probabilities[key] = round(value / total, 3)
        else:
            print("Probabilities histogram is empty, cannot normalize.")

    def initialize_uniform(self):
        """
        Initializes a uniform distribution.

        :return: None
        """
        if self.raw_values:
            n = len(self.raw_values)
            for key in self.raw_values.keys():
                self.raw_values[key] = 1.0 / n

    def add(self, object):
        """
        Adds an object to the set.

        :param object: ObjectFrame item
        :return: None
        """
        p = self.p(object)
        #print("{0} ==> {1}".format(object,p))
        self.raw_values[object.name] = p
        #self.normalize_all()

    def print_probabilities(self):
        """
        Prints everything.

        :return: None
        """

        for key, value in self.normalized_probabilities.items():
            print("{0}: {1}%".format(key, round(value*100.0, 2)))
        print("\n#-------#")

    def save_probabilities(self, timestep):
        """
        Saves the current probabilities.

        :param timestep: current timestep
        """
        with open(self.log, 'a') as f:
            writer = csv.writer(f)
            row = list(self.normalized_probabilities.values())
            row.insert(0, timestep)
            writer.writerow(row)

    def get_ranked_probabilities(self):
        """
        Sorts the objects based on their probability.

        :return: List of tuples (object, probability), ordered from the highest to the lowest probability.
        """
        d = {k: v for k, v in sorted(self.normalized_probabilities.items(), key=lambda item: item[1], reverse=True)}
        return list(d.items())

    def get_top_n_items(self, n=0):
        """
        Returns the top N items of the ranked probabilities. When n=0, returns everything (equivalent to
        get_ranked_probabilities()). If more items are requested than the number of items available, it will still
        fulfill the request by appending (None, 0.0) values to the returned list.

        :param n: Number of elements to return
        :return: Top N ranked probabilities
        """
        assert n >= 0, "Requirement: n >= 0."
        ranked_prob = self.get_ranked_probabilities()
        if n == 0:      # No limit it set: let's return all the results
            output = ranked_prob
        elif n > len(ranked_prob):  # More items have been requested than the ones available: fill in the missing data
            extension = []
            missing = (None, 0.0)
            missing_no = n - len(ranked_prob)
            for i in range(missing_no):
                extension.append(missing)
            output = ranked_prob.extend(extension)
        else:       # Return exactly the first N items
            output = ranked_prob[:n]
        return [self.FocusItem(x[0], x[1]) for x in output]

    def process_iteration(self):
        """
        After having accepted all the objects for the current iteration, it normalizes the probabilities and calculates
        the existence of a clear winner of the focus estimation.

        @return: None
        """
        self.normalize_all()
        top_items = self.get_top_n_items(n=3)
        item1 = top_items[0]
        item2 = top_items[1]
        item3 = top_items[2]    # The third item is used only to verify if the second is actually dominant
        if item1.p == item2.p == item3.p:
            # Three ties: can't really tell what's going on
            # If there is already a recorded target, keep it, otherwise ignore
            if self.target is not None:
                item_list = [item1.name, item2.name, item3.name]
                target, non_targets = self.target_window.contains_winner(item_list)
                # The previously defined target should remain so
                self.target_window.add(target)
                for non_target in non_targets:
                    # Add the remaining items as destination candidates
                    self.destination_window.add(non_target)
        elif item1.p == item2.p:
            # Tie between first and second ranked elements
            # If there is already a recorded target then keep it, otherwise add them to the window
            if self.target is not None:
                item_list = [item1.name, item2.name]
                target, non_target = self.target_window.contains_winner(item_list)
                # The previously defined target should remain so
                self.target_window.add(target)
                # Add the remaining item as destination candidates
                self.destination_window.add(non_target[0])
            else:
                # If there is still no clear winner, add them both to the competition
                self.target_window.add(item1.name)
                self.target_window.add(item2.name)
                # Maybe the third elements can shed light on the destination?
                self.destination_window.add(item3.name)
        else:
            # No tie. First element is dominant, second is a possible destination, third is ignored
            if item1.p >= self.threshold - self.epsilon:
                # These elements are recorded only if the highest ranking surpasses the threshold
                self.target_window.add(item1.name)
                if item2.p > 0.05:
                    self.destination_window.add(item2.name)
        # At this point, check if there are clear winners for target and destination
        if self.target_window.check_winner():
            self.target = self.target_window.most_recent_winner
        if self.destination_window.check_winner():
            self.destination = self.destination_window.most_recent_winner
        # Sanity check: I can't have a destination without a target
        if self.target is None:
            self.destination = None

    def get_winners_if_exist(self):
        """
        Returns the predicted target and destination, if available.

        @return: target, destination (names, can be None)
        """
        return self.target, self.destination
