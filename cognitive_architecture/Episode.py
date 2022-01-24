"""
These three classes model an episode.
Episode contains a set of HumanFrames, which can interact with several ObjectFrames.
"""
import numpy as np
import pandas as pd


class Episode:
    def __init__(self, time=0, humans=None):
        if humans is None:
            humans = {}
        # Sanity check
        assert isinstance(time, int) and time >= 0, "Time must be a positive integer, provided: {0}".format(time)
        self.time = time
        self.humans = humans

    def get_objects_for(self, human_name):
        """
        Retrieves the objects with which the specified human is interacting during this episode.

        :param human_name: str
        :return: list of ObjectFrame
        """
        if self.humans:
            hf = self.humans.get(human_name)
            if hf is not None:
                return [object for object in hf.objects.values()]
            else:
                return None

    def to_feature(self, human, train=True):
        """
        Returns a feature vector for a specific human, if found.

        :param human: str, name of the human.
        :param train: if True, creates a training feature, otherwise a test one.
        :return: 1x5 feature array, or None if not found.
        """
        try:
            hf = self.humans[human]
            if train:
                feature = hf.to_train_feature()
                feature.insert(0, self.time)
            else:
                feature = hf.to_test_feature()
            return feature
        except KeyError:
            return None

    def __str__(self):
        #return "{0}: {1}".format(self.time, self.humans['human'].get_label())
        #return "{0}: {1}".format(self.time, self.humans['human'].to_test_feature())
        hf = self.humans['human']
        return "Episode at time {0} [target {1}]: {2}".format(self.time, hf.target, hf)


class HumanFrame:
    def __init__(self, name, mos='s', hold=False, objects=None, target=None, fallback_label=None, x=0, y=0, ov=0.0):
        if objects is None:
            objects = {}
        # Sanity checks
        assert isinstance(mos, str), "MOS must be of type string."
        assert mos.upper() == "M" or mos.upper() == "S", "MOS must have value 'm' or 's'."
        assert isinstance(hold, bool), "HOLD must be True or False."
        self.name = name
        self.MOS = mos.upper()
        self.HOLD = hold
        self.objects = objects
        self.target = target
        self.fallback_label = fallback_label
        self.x = x
        self.y = y
        self.ov = ov

    def is_stationary(self):
        """
        Is the human stationary? Equivalent to not is_moving().

        :return: True/False
        """
        return self.MOS == 'S'

    def is_moving(self):
        """
        Is the human moving? Equivalent to not is_stationary().

        :return: True/False
        """
        return self.MOS == 'M'

    def is_holding(self):
        """
        Is the human holding an item?

        :return: True/False
        """
        return self.HOLD is True

    def get_object_of_interest(self):
        """
        Used during training, this method retrieves the object which has a label attached, if it exists.

        :return: ObjectFrame or None
        """
        for object in self.objects.values():
            if object.label is not None:
                return object
        return None

    def to_train_feature(self):
        """
        Builds a feature vector out of the current episode's data.

        :return: 1x5 feature vector [MOS, HOLD, QDC, QTC, LABEL]
        """
        mos = True if self.MOS == 'M' else False
        hold = bool(self.HOLD)
        ooi = self.get_object_of_interest()
        if ooi is None:
            return [mos, hold, 'IGNORE', 'IGNORE', self.fallback_label]
        else:
            return [mos, hold, ooi.QDC, ooi.QTC, ooi.label]

    def to_test_feature(self, nparray=True):
        """
        Builds a (test) feature vector out of the current episode's data.

        :param nparray: if True, will return a numpy array instead of a simple list
        :return: 1x4 feature vector [MOS, HOLD, QDC, QTC]
        """
        mos = True if self.MOS == 'M' else False
        hold = bool(self.HOLD)
        if self.target is None:
            feature_list = [mos, hold, 'IGNORE', 'IGNORE']
        else:
            try:
                ooi = self.objects[self.target]
                feature_list = [mos, hold, ooi.QDC, ooi.QTC]
            except KeyError:
                print("Invalid target '{0}', defaulting to None.".format(self.target))
                feature_list = [mos, hold, 'IGNORE', 'IGNORE']
        if not nparray:
            return feature_list
        else:
            # Convert the feature list to a numpy array
            qdc_mapping = {
                'TOUCH': 1,
                'NEAR': 2,
                'MEDIUM': 3,
                'FAR': 4,
                'IGNORE': 5
            }
            qtc_mapping = {
                '0': 1,
                '-': 2,
                '+': 3,
                'IGNORE': 4
            }
            df = pd.DataFrame(
                data={
                    'MOS': np.array([feature_list[0]], dtype=bool),
                    'HOLD': np.array([feature_list[1]], dtype=bool),
                    'QDC': np.array([qdc_mapping[feature_list[2]]], dtype=int),
                    'QTC': np.array([qtc_mapping[feature_list[3]]], dtype=int),
                }
            )
            return df.to_numpy()

    def get_label(self):
        """
        Retrieves the training label (which might be the fallback one).

        :return: The training label, str
        """
        object_of_interest = self.get_object_of_interest()
        if object_of_interest is not None:
            return object_of_interest.label
        else:
            return self.fallback_label

    def get_position(self):
        """
        Returns the (x,y) coordinate.

        :return: Coordinate list [x,y]
        """
        return [self.x, self.y]

    def __str__(self):
        if self.target:
            try:
                of = self.objects[self.target]
                return "[MOS: {0}, HOLD: {1}, QDC: {2}, QTC: {3}], ov = {4}".format(self.MOS, self.HOLD, of.QDC, of.QTC, self.ov)
            except KeyError:
                return "[No data to display]"
        else:
            return "[MOS: {0}, HOLD: {1}, QDC: unknown, QTC: unknown]".format(self.MOS, self.HOLD)


class ObjectFrame:
    def __init__(self, name, qdc, qtc, label=None, x=0, y=0, theta=0):
        # Sanity checks
        assert isinstance(qdc, str), "QDC must be of type string."
        assert qtc in ['-', '0', '+'], "QTC must be one of -, 0 or +, as string."
        self.name = name
        self.QDC = qdc.upper()
        self.QTC = qtc
        self.label = label
        self.x = x
        self.y = y
        self.theta = theta

    def __str__(self):
        return "{0}: QDC: {1}, QTC: {2}, angle: {3}".format(self.name, self.QDC, self.QTC, self.theta)
