"""
Cognitive Architecture
"""

from cognitive_architecture.LowLevel import LowLevel
from cognitive_architecture.HighLevel import HighLevel
from threading import Thread
from cognitive_architecture.ObservationLibrary import ObservationLibrary
from cognitive_architecture.InternalComms import InternalComms
from cognitive_architecture.Bridge import Bridge

DOMAIN_FILE = "Domain_kitchen.xml"


class CognitiveArchitecture(Thread):
    def __init__(self, mode):
        self.mode = mode.upper()
        assert self.mode == "TRAIN" or mode == "TEST", "mode accepts parameters 'train' or 'test'."
        Thread.__init__(self)
        self.internal_comms = InternalComms()
        #self.tq = ObservationQueue()
        self.tq = ObservationLibrary()
        self.lowlevel = LowLevel(self.tq, self.internal_comms) # todo tq only inside LowLevel?
        self.highlevel = HighLevel(self.internal_comms, DOMAIN_FILE)    # No observations are provided on startup

    def is_trainingmode(self):
        return True if self.mode == "TRAIN" else False

    def run(self):
        print("[DEBUG] " + self.__class__.__name__ + " thread is running in {0} mode.\n".format(self.mode))
        if self.mode == "TRAIN":
            self.lowlevel.train(min=0, max=0, save_id=None)
        else:   # TEST
            self.lowlevel.load()    # Reloads the trained models
            self.highlevel.start()
            self.lowlevel.test()
