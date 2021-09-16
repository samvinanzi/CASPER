"""

An extension of standard Thread that implements a flagging mechanism to signal the end of the execution.

"""

from threading import Thread


class StopThread(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.stop_flag = False

    # Flags the stop variable
    def stop(self):
        self.stop_flag = True
