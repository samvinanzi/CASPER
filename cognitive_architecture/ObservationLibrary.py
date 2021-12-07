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

        def compute_qsr(self, show=False):
            def pretty_print_world_qsr_trace(qsrlib_response_message):
                """
                Just a print function.

                :param qsrlib_response_message: QSRLib response message
                :return: None
                """
                print(self.which_qsr, "request was made at ", str(qsrlib_response_message.req_made_at)
                      + " and received at " + str(qsrlib_response_message.req_received_at)
                      + " and finished at " + str(qsrlib_response_message.req_finished_at))
                print("---")
                print("Response is:")
                for t in qsrlib_response_message.qsrs.get_sorted_timestamps():
                    foo = str(t) + ": "
                    for k, v in zip(qsrlib_response_message.qsrs.trace[t].qsrs.keys(),
                                    qsrlib_response_message.qsrs.trace[t].qsrs.values()):
                        foo += str(k) + ":" + str(v.qsr) + "; "
                    print(foo)

            qsrlib_request_message = QSRlib_Request_Message(self.which_qsr, self.world_trace, dynamic_args=self.dynamic_args)
            qsrlib_response_message = self.qsrlib.request_qsrs(req_msg=qsrlib_request_message)
            if show:
                pretty_print_world_qsr_trace(qsrlib_response_message)
            return qsrlib_response_message, self.world_trace

    def __init__(self, debug=False):
        self.debug = debug
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
            self.observations.extend(observation)
            if self.debug:
                print("[OBSLIB] Producer inserted observation")
        self.observations_event.set()

    def process_observations(self, debug=True):
        """
        Thread that runs in background. As soon as new observartions are available, it produces a new set of QSRs and
        makes them available for the consumer.

        :return: None
        """
        while True:
            # Wait for an observation to happen
            if self.debug:
                print("[OBSLIB] Waiting for a producer event...")
            self.observations_event.wait()
            # Retrieve all the data
            with self.observations_lock:
                if self.debug:
                    print("[OBSLIB] Producer event detected! Fetching observations.")
                observations = self.observations
            self.observations_event.clear()
            # Calculate the QSRs
            self.qsr_parser.world_trace.add_object_state_series(observations)
            if not len(self.qsr_parser.world_trace.get_sorted_timestamps()) >= 2:
                # At least 2 timestamps are required in order to calculate the QSRs
                continue
            else:
                qsr = self.qsr_parser.compute_qsr(show=False)
                # Store the QSR library for the consumer
                with self.qsr_lock:
                    if self.debug:
                        print("[OBSLIB] Updating the QSR Library with timestamp {0}.".format(
                            len(self.qsr_parser.world_trace.get_sorted_timestamps())))
                    self.qsr = qsr
                self.qsr_event.set()

    def retrieve_qsrs(self):
        """
        Method called by the consumer, which retrieves the latest QSRs.

        :return: QRS response message
        """
        self.qsr_event.wait()
        if self.debug:
            print("[OBSLIB] Consumer accessing library.")
        with self.qsr_lock:
            qsr = self.qsr
        self.qsr_event.clear()
        return qsr
