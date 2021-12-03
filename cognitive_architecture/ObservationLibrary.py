"""
This class is activated by CognitiveArchitecture and is a middle-ground between the robot (producer) and the cognitive
architecture (consumer). Visual observations are inserted and an up-to-date QSR library is maintained for the consumer
to access whenever necessary.

By providing the consumer with the global set of QSRs, instead of the newest observation, latency problems between the
two entities is addressed.
"""

from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from qsrlib_io.world_trace import World_Trace
from threading import Thread, Event, Lock


class ObservationLibrary:
    class QSRParser:
        """
        This utility class parses the QSR Request Message.
        """
        def __init__(self):
            self.qsrlib = QSRlib()
            self.world_trace: World_Trace = World_Trace()
            self.which_qsr = ["argd", "qtcbs", "mos"]
            objects_of_interest = ["sink", "glass", "hobs", "biscuits", "meal", "plate", "bottle"]  # todo external config?
            qsrs_for = []
            for ooi in objects_of_interest:
                qsrs_for.append(("human", ooi))
            self.dynamic_args = {
                "argd": {
                    "qsrs_for": qsrs_for,
                    "qsr_relations_and_values": {"touch": 0.6, "near": 2, "medium": 3, "far": 5}
                },
                "qtcbs": {
                    "qsrs_for": qsrs_for,
                    "quantisation_factor": 0.01,
                    "validate": False,
                    "no_collapse": True
                },
                "mos": {
                    "qsr_for": ["human"],
                    "quantisation_factor": 0.09
                }
            }

        def compute_qsr(self):
            qsrlib_request_message = QSRlib_Request_Message(self.which_qsr, self.world_trace, dynamic_args=self.dynamic_args)
            qsrlib_response_message = self.qsrlib.request_qsrs(req_msg=qsrlib_request_message)
            return qsrlib_response_message

    def __init__(self):
        self.qsr_parser = self.QSRParser()
        # Protected variables
        self.observations = []
        self.qsr = None
        # Synchronization entities
        self.observations_event = Event()
        self.observations_lock = Lock()
        self.qsr_event = Event()
        self.qsr_lock = Lock()
        # Initializes the processing thread
        t = Thread(target=self.process_observations)
        t.start()

    def add_observation(self, observation):
        """
        Method called by the producer, which stores an observation in the set.

        :param observation: list of Object_State
        :return: None
        """
        with self.observations_lock:
            self.observations.append(observation)
        self.observations_event.set()

    def process_observations(self):
        """
        Thread that runs in background. As soon as new observartions are available, it produces a new set of QSRs and
        makes them available for the consumer.

        :return: None
        """
        while True:
            # Wait for an observation to happen
            self.observations_event.wait()
            # Retrieve all the data
            with self.observations_lock:
                observations = self.observations
            self.observations_event.clear()
            # Calculate the QSRs
            self.qsr_parser.world_trace.add_object_state_series(observations)
            qsr = self.qsr_parser.compute_qsr()
            # Store the QSR library for the consumer
            with self.qsr_lock:
                self.qsr = qsr
            self.qsr_event.set()

    def retrieve_qsrs(self):
        """
        Method called by the consumer, which retrieves the latest QSRs.

        :return: QRS response message
        """
        self.qsr_event.wait()
        with self.qsr_lock:
            qsr = self.qsr
        self.qsr_event.clear()
        return qsr
