"""
Cognitive Architecture
"""

from cognitive_architecture.LowLevel import LowLevel
from cognitive_architecture.HighLevel import HighLevel
from threading import Thread
from cognitive_architecture.ObservationQueue import ObservationQueue
from cognitive_architecture.InternalComms import InternalComms
from cognitive_architecture.Bridge import Bridge

DOMAIN_FILE = "data/CRADLE/Domain_kitchen.xml"
OBSERVATION_FILE = "data/CRADLE/Observations_kitchen.xml"   # todo change to dynamically generated XML file


class CognitiveArchitecture(Thread):
    def __init__(self, mode):
        self.mode = mode.upper()
        assert self.mode == "TRAIN" or mode == "TEST", "mode accepts parameters 'train' or 'test'."
        Thread.__init__(self)
        self.tq = ObservationQueue()
        self.internal_comms = InternalComms()
        self.lowlevel = LowLevel(self.tq, self.internal_comms) # todo tq only inside LowLevel?
        self.highlevel = HighLevel(self.internal_comms, DOMAIN_FILE)    # No observations are provided on startup

    def is_trainingmode(self):
        return True if self.mode == "TRAIN" else False

    def run(self):
        if self.mode == "TRAIN":
            self.lowlevel.train(min=0, max=0, save_id=None)
        else:   # TEST
            self.lowlevel.load()    # Reloads the trained models
            self.highlevel.run()
            self.lowlevel.test()
