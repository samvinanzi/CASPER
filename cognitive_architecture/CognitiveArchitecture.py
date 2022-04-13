"""
Cognitive Architecture
"""

from cognitive_architecture.LowLevel import LowLevel
from cognitive_architecture.HighLevel import HighLevel
from threading import Thread
from cognitive_architecture.InternalComms import InternalComms

DOMAIN_FILE = "Domain_kitchen_corrected.xml"


class CognitiveArchitecture(Thread):
    def __init__(self, mode):
        self.mode = mode.upper()
        assert self.mode == "TRAIN" or mode == "TEST", "mode accepts parameters 'train' or 'test'."
        Thread.__init__(self)
        self.internal_comms = InternalComms()
        self.lowlevel = LowLevel(self.internal_comms)
        self.highlevel = HighLevel(self.internal_comms, DOMAIN_FILE, debug=True)    # No observations given on startup

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
