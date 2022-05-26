"""
Wrapper for the synchronization objects.
"""

from multiprocessing import Lock, Event, Queue
import pickle
from util.PathProvider import path_provider


class QSRSynch:
    """
    Used to manipulate the shared QSR library which is stored in the local drive through serialization.
    """
    def __init__(self):
        self.lock = Lock()
        self.event = Event()

    def set(self, qsr):
        with self.lock:
            pickle.dump(qsr, open(path_provider.get_pickle('qsr.p'), "wb"))
        self.event.set()

    def get(self):
        self.event.wait()
        with self.lock:
            qsr = pickle.load(open(path_provider.get_pickle('qsr.p'), "rb"))
        self.event.clear()
        return qsr


class SynchVariable:
    """
    A generic synchronization object used to share data between processes.
    """
    def __init__(self):
        self.value = Queue()
        self.lock = Lock()
        self.event = Event()

    def set(self, value):
        with self.lock:
            self.value.put(value)
        self.event.set()

    def get(self):
        if self.event.wait():
            with self.lock:
                value = self.value.get()
            self.event.clear()
            return value
        else:
            return None

    def poll(self):
        return self.event.is_set()
