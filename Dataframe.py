class Episode:
    def __init__(self, time=0, humans=None):
        if humans is None:
            humans = {}
        # Sanity check
        assert isinstance(time, int) and time >= 0, "Time must be a positive integer."
        self.time = time
        self.humans = humans


class HumanFrame:
    def __init__(self, mos='s', hold=False, objects=None, fallback_label=None):
        if objects is None:
            objects = {}
        # Sanity checks
        assert isinstance(mos, str), "MOS must be of type string."
        assert mos.upper() == "M" or mos.upper() == "S", "MOS must have value 'm' or 's'."
        assert isinstance(hold, bool), "HOLD must be True or False."
        self.MOS = mos.upper()
        self.HOLD = hold
        self.objects = objects
        self.fallback_label = fallback_label

    def is_stationary(self):
        return self.MOS == 'S'

    def is_moving(self):
        return self.MOS == 'M'

    def is_holding(self):
        return self.HOLD == True

    def to_feature(self):
        features = []
        if not self.objects:
            # no objects found
            feature = [self.MOS, self.HOLD, None, None, self.fallback_label]
            features.append(feature)
        else:
            for name, qsrs in self.objects.items():
                feature = [self.MOS, self.HOLD, qsrs.QDC, qsrs.QTC, qsrs.label]
                features.append(feature)
        return features


class ObjectFrame:
    def __init__(self, qdc, qtc, label=None):
        # Sanity checks
        assert isinstance(qdc, str), "QDC must be of type string."
        assert qtc in ['-', '0', '+'], "QTC must be one of -, 0 or +, as string."
        self.QDC = qdc.upper()
        self.QTC = qtc
        self.label = label

'''
ep = Episode(time=0, humans={'Peppe': HumanFrame(mos='m', hold=False, objects={
    'coca-cola': ObjectFrame(qdc='Near', qtc='0'),
    'table(1)': ObjectFrame(qdc='Near', qtc='0')
    })})

f = ep.humans['Peppe'].to_feature()
pass
'''