"""
Cognitive Architecture
"""

from cognitive_architecture.LowLevel import LowLevel
from cognitive_architecture.HighLevel import HighLevel
from threading import Thread
from cognitive_architecture.ObservationQueue import ObservationQueue

DOMAIN_FILE = "data/CRADLE/Domain_kitchen.xml"
OBSERVATION_FILE = "data/CRADLE/Observations_kitchen.xml"   # todo change to dynamically generated XML file


class CognitiveArchitecture(Thread):
    def __init__(self, mode):
        self.mode = mode.upper()
        assert self.mode == "TRAIN" or mode == "TEST", "mode accepts parameters 'train' or 'test'."
        Thread.__init__(self)
        self.tq = ObservationQueue()
        self.lowlevel = LowLevel(self.tq)
        self.highlevel = HighLevel()

    def is_trainingmode(self):
        return True if self.mode == "TRAIN" else False

    def run(self):
        if self.mode == "TRAIN":
            self.lowlevel.train(min=0, max=0, save_id=None)
        else:   # TEST
            self.lowlevel.load()    # Reloads the trained models
            self.lowlevel.test()
