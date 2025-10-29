"""
Cognitive Architecture
"""

from cognitive_architecture.LowLevel import LowLevel
from cognitive_architecture.HighLevel import HighLevel
from multiprocessing import Process
from datatypes.Synchronization import SynchVariable


class CognitiveArchitecture(Process):
    def __init__(self, robot_conn, qsr_synch, start_event, mode, verification=True):
        super().__init__()
        assert mode.upper() in ['TRAIN', 'TEST'], "mode accepted values: 'train', 'test'."
        self.robot_conn = robot_conn
        self.start_event = start_event
        self.mode = mode
        # Creates synchronized objects for LowLevel and HighLevel and initializes them
        self.obs_from_ll_conn = SynchVariable()
        self.lowlevel = LowLevel(self.obs_from_ll_conn, qsr_synch, mode, verification)
        #print("Training Decision Tree...")
        #self.lowlevel.train_decision_tree()
        #self.lowlevel.save()
        #print("Decision Tree trained.")
        self.obs_to_hl_conn = SynchVariable()
        self.goal_from_hl_conn = SynchVariable()
        self.highlevel = HighLevel(self.obs_to_hl_conn, self.goal_from_hl_conn, verification)

    def run(self) -> None:
        print("{0} process is running in {1} mode.".format(self.__class__.__name__, self.mode))
        # Instructs the main process to start the subprocesses (LL and HL)
        self.start_event.set()
        if not self.is_training():
            while True:
                # HL has the priority: check if a goal has been found
                if self.goal_from_hl_conn.poll():
                    plan = self.goal_from_hl_conn.get()
                    # Send the collaborative instructions to the Robot
                    self.robot_conn.set(plan)
                elif self.obs_from_ll_conn.poll():
                    # If HL is silent, but LL has a new observation, add it to the poll
                    observation = self.obs_from_ll_conn.get()
                    self.obs_to_hl_conn.set(observation)
        else:
            self.lowlevel.start()

    def is_training(self):
        return True if self.mode.upper() == "TRAIN" else False
