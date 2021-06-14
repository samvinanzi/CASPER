"""
These three classes model an episode.
Episode contains a set of HumanFrames, which can interact with several ObjectFrames.
"""


class Episode:
    def __init__(self, time=0, humans=None):
        if humans is None:
            humans = {}
        # Sanity check
        assert isinstance(time, int) and time >= 0, "Time must be a positive integer."
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

    def __str__(self):
        return "{0}: {1}".format(self.time, self.humans['human'].get_label())


class HumanFrame:
    def __init__(self, name, mos='s', hold=False, objects=None, fallback_label=None):
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
        self.fallback_label = fallback_label

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

    def to_feature(self):
        """
        Builds a feature vector out of the current episode's data.

        :return: 1x5 feature vector [MOS, HOLD, QDC, QTC, LABEL]
        """
        ooi = self.get_object_of_interest()
        if ooi is None:
            return [self.MOS, self.HOLD, None, None, self.fallback_label]
        else:
            return [self.MOS, self.HOLD, ooi.QDC, ooi.QTC, ooi.label]

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


class ObjectFrame:
    def __init__(self, name, qdc, qtc, label=None):
        # Sanity checks
        assert isinstance(qdc, str), "QDC must be of type string."
        assert qtc in ['-', '0', '+'], "QTC must be one of -, 0 or +, as string."
        self.name = name
        self.QDC = qdc.upper()
        self.QTC = qtc
        self.label = label
