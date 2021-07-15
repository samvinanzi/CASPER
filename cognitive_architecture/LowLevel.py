"""
Low-Level cognitive architecture (from QSR to contextualized actions)
"""
import numpy as np

from cognitive_architecture.FocusBelief import FocusBelief
from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from qsrlib_io.world_trace import World_Trace
import pickle
import os
from pathlib import Path
from cognitive_architecture.TreeTrainer import TreeTrainer
from cognitive_architecture.EpisodeFactory import EpisodeFactory
from cognitive_architecture.MarkovFSM import ensemble
from cognitive_architecture.Contextualizer import Contextualizer

BASEDIR = basedir = Path(__file__).parent.parent
PICKLE_DIR = "data/pickle"
SAVE_DIR = "data/cognition"


class LowLevel:
    def __init__(self, tq):
        self.tq = tq
        self.world_trace = World_Trace()
        self.qsrlib = QSRlib()
        self.which_qsr = ["argd", "qtcbs", "mos"]
        self.tree = None
        self.focus = FocusBelief(human_name="human")
        self.ensemble = ensemble

        self.dynamic_args = {
            "argd": {
                "qsrs_for": [("human", "coca-cola"), ("human", "table(1)")],
                "qsr_relations_and_values": {"touch": 0.6, "near": 1, "medium": 3, "far": 5}
            },
            "qtcbs": {
                "qsrs_for": [("human", "coca-cola"), ("human", "table(1)")],
                "quantisation_factor": 0.01,
                "validate": False,
                "no_collapse": True
            },
            "mos": {
                "qsr_for": ["human"],
                "quantisation_factor": 0.09
            }
        }

    def save(self):
        """
        Saves the cognitive architecture.

        :return: None
        """
        path = (BASEDIR / SAVE_DIR / "tree.p").resolve()
        pickle.dump(self.tree, open(path, "wb"))

    def load(self):
        """
        Loads the cognitive architecture.

        :return: None
        """
        path = (BASEDIR / SAVE_DIR / "tree.p").resolve()
        self.tree = pickle.load(open(path, "rb"))

    def observe(self, to_observe=float('inf')):
        """
        Observes a certain number of timesteps, collects the ObjectStates and adds them to the world trace.

        :param to_observe: number of timesteps to observe
        :return: The timestamp of the latest observation
        """
        assert to_observe > 0, "Must request observations for at least 1 timeframe."
        n = 0
        current_timestamp = 0
        while True:
            observations = self.tq.get()    # Blocking call
            if observations is not None:
                for observation in observations:
                    self.world_trace.add_object_state(observation)
                    current_timestamp = int(observation.timestamp)
                n += 1
                if n >= to_observe:
                    return current_timestamp

    def compute_qsr(self, save_id=None, show=False):
        """
        Computes the QSR from the world trace.

        :param save_id: id of the training sub-set.
        :param show: if True, prints the output
        :return: World_QSR_Trace
        """
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
        if save_id is not None:
            assert isinstance(save_id, int), "save_id must be of type int."
            base_pickle_dir = os.path.join(BASEDIR, PICKLE_DIR)
            pickle.dump(qsrlib_response_message, open(os.path.join(base_pickle_dir,
                                                                   "qsr_response{0}.p".format(save_id)),"wb"))
            pickle.dump(self.world_trace, open(os.path.join(base_pickle_dir, "world_trace{0}.p".format(save_id)), "wb"))
            print("[DEBUG] Pickled QSR data to {0}".format(base_pickle_dir))
        return qsrlib_response_message

    def train(self, min=0, max=9, save_id=None):
        """
        Instantiates the chain of training processing. Must be executed in steps: first build N training sub-sets, then
        combine them and use them to train.

        :param min: Initial range of the datasets to consider
        :param max: Final range of the datasets to consider
        :param save_id: id of the training sub-set
        :return: None
        """
        self.observe(40)
        self.compute_qsr(save_id=None, show=True)
        # Re-enable these lines after all training sets have been saved to combine them and train the tree.
        # It *could* be automated with the addition of a 'scene master' in the Webots scenery.
        '''
        self.prepare_trainingset(min, max, kfold=True)
        score = self.k_fold_validation(min, max)
        self.train_decision_tree()
        '''

    def prepare_trainingset(self, min=0, max=9, kfold=False):
        """
        To invoke once N training sub-sets have been produced. Loads the pickles, generates the Episodes, exports the
        datasets as CSV and combines them in an 'all.csv' training set file.

        :param min: Initial range of the datasets to consider
        :param max: Final range of the datasets to consider
        :param kfold: if True, performs 1-fold cross-validation
        :return: None
        """
        trainer = TreeTrainer()
        trainer.prepare_datasets(min, max)
        trainer.combine_datasets(min, max)
        if kfold:
            trainer.create_k_folds(min, max)

    def train_decision_tree(self):
        """
        Trains the Decision Tree based on the 'all.csv' training file.

        :return: None
        """
        trainer = TreeTrainer()
        self.tree = trainer.train_model('all.csv')

    def k_fold_validation(self, min=0, max=9):
        """
        Performs 1-fold cross-validation.

        :param min: Initial range of the datasets to consider
        :param max: Final range of the datasets to consider
        :return: score
        """
        trainer = TreeTrainer()
        return trainer.k_fold_cross_validation(min, max)

    def test(self, debug=True):
        while True:
            # Collect the latest observation
            latest_timestamp = self.observe(1)
            if debug:
                print("#----------- TIME {0} -----------#".format(latest_timestamp))
            # QSRs can be computed only if there are at least 2 timestamps in the world trace
            if not len(self.world_trace.get_sorted_timestamps()) >= 3:      # We work with T-1 (see below)
                if debug:
                    print("Not enought data, continuing to observe...")
                continue
            else:
                # Process it into a set of QSRs
                qsr_response = self.compute_qsr()
                factory = EpisodeFactory(self.world_trace, qsr_response)
                # QTC is only calculated at step T-1, so we work with that one
                episode = factory.build_episode(latest_timestamp-1)
                print("Episode: {0}".format(episode))
                if episode is None:
                    if debug:
                        print("No human found in this frame. Continuing to observe...")
                    continue
                else:
                    # Assess the focus of the human to identify the object of interest
                    objects_in_timestep = episode.get_objects_for("human")
                    for object in objects_in_timestep:
                        self.focus.add(object)
                    if not self.focus.has_confident_prediction():
                        if debug:
                            print("No clear focus yet, observing...")
                        continue    # If no confident focus predictions were made, we need to observe more
                    else:
                        # We have a target: add it to the episode and generate a feature
                        target = self.focus.get_top_n_items(1)
                        target_name, target_score = list(target.items())[0]
                        if debug:
                            print("Target identified: {0} ({1}%)".format(target_name, target_score))
                        episode.humans["human"].target = target_name
                        feature = episode.to_feature(human="human", train=False)
                        # Classify the target's set of QSRs into a Movement
                        movement = self.tree.predict(feature)[0]
                        if debug:
                            print("Predicted movement: {0}".format(movement))
                        # Add the Movement to the markovian finite-state machine to predict a temporal Action
                        ensemble.add_observation(movement)
                        action, score, winner = ensemble.best_model()
                        if debug:
                            print("Best action fit: {0} ({1}). Winner... {2}".format(action, score, winner))
                        if not winner:
                            continue    # More data is needed
                        else:
                            #ensemble.empty_observations()   # Reset the observations
                            # Contextualize the Action
                            contextualizer = Contextualizer()
                            # The second item of the focus is the destination
                            context = list(self.focus.get_top_n_items(2))[1]
                            ca = contextualizer.give_context(action, context)
                            if debug:
                                print("{0} was contextualized in {1}".format(action, ca))
                            # Try to predict the overall plan
                            # todo
                            break
                    # Collaborate
                    # todo
