from queue import Queue
from asyncio import QueueEmpty, QueueFull


class ObservationQueue:
    def __init__(self):
        self.queue = Queue()

    def put(self, item):
        try:
            self.queue.put_nowait(item)    # does not block if queue is full (to avoid losing future data)
        except QueueFull:
            print("[ERROR] Queue full, unable to fulfill latest insertion request")

    def get(self):
        try:
            item = self.queue.get()
            self.queue.task_done()
            return item
        except QueueEmpty:
            return None
