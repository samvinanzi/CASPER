"""
This class models a low/high -level communication queue made up by two synchronized elements: a Queue and an Event.
"""

from queue import Queue
from asyncio import QueueEmpty, QueueFull
from threading import Event, Lock


class InternalComms:
    def __init__(self):
        self.queue = Queue()            # contains the cluster transitions
        self.queue_event = Event()      # signals the presence of new data in vocal_queue
        self.name_event = Event()       # signals the presence of new data in goal_name
        self.lock = Lock()              # to protect goal_name
        self.goal_name = None           # inferred goal name
        self.frontier = None            # inferred plan frontier

    # Inserts an item in the queue and signals the event
    def put(self, item):
        try:
            self.queue.put(item)
        except QueueFull:
            print("[ERROR] Queue full, unable to fulfil latest insertion request")
        self.queue_event.set()

    # Waits for an event to occur, then tries to retrieve the item and signals the operation as completed
    def get(self):
        self.queue_event.wait()
        if not self.queue.empty():
            try:
                item = self.queue.get()
                self.queue.task_done()
                self.queue_event.clear()
                return item
            except QueueEmpty:
                print("[ERROR] Invalid access to empty transition queue")

    # Writes the inferred goal name
    def write_goal(self, name, frontier):
        assert name is not None, "invalid goal name"
        with self.lock:
            self.goal_name = name
            self.frontier = frontier
        self.name_event.set()

    # Retrieves the goal name
    def get_goal(self):
        self.name_event.wait()
        with self.lock:
            name = self.goal_name
            frontier = self.frontier
        self.name_event.clear()
        return name, frontier

    # Has a goal name been found?
    def was_goal_inferred(self):
        with self.lock:
            name = self.goal_name
        return True if name else False
