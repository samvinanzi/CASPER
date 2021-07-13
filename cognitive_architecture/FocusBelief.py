"""
Models the robot's belief on the human's focus.
"""

from cognitive_architecture.Episode import ObjectFrame


class FocusBelief:
    def __init__(self, human_name, w_qdc=1, w_qtc=1):
        self.name = human_name
        self.w_qdc = w_qdc
        self.w_qtc = w_qtc
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
        return self.w_qdc * self.qdc_map[object.QDC] + self.w_qtc * self.qtc_map[object.QTC]

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
        self.raw_values[object.name] = p
        self.normalize_all()

    def print_probabilities(self):
        """
        Prints everything.

        :return: None
        """
        for key, value in self.normalized_probabilities.items():
            print("{0}: {1}%".format(key, value))
        print("\n#-------#")

    def get_ranked_probabilities(self):
        """
        Sorts the objects based on their probability.

        :return: Ordered dictionary, with most probable items on the top.
        """
        return {k: v for k, v in sorted(self.normalized_probabilities.items(), key=lambda item: item[1], reverse=True)}

    def get_top_n_items(self, n=0):
        """
        Returns the top N items of the ranked probabilities. When n=0, returns everything (equivalent to
        get_ranked_probabilities()).

        :param n: Number of elements to return
        :return: Top N ranked probabilities
        """
        assert n >= 0, "Requirement: n >= 0."
        ranked_prob = self.get_ranked_probabilities()
        if n == 0:
            return ranked_prob
        else:
            return dict(list(ranked_prob.items())[:n])

    def has_confident_prediction(self, threshold=.6):
        """
        Determines if there is one confident prediction.

        :return: True or False.
        """
        assert .5 < threshold <= 1.0, "Threshold should be at least .51, max 1.0"
        top_item = self.get_top_n_items(1)
        if list(top_item.values())[0] >= threshold:
            return True
        else:
            return False
