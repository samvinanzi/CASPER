"""

Utility to log the execution parameters of the cognitive architecture.

"""

import csv
from util.PathProvider import path_provider
import os.path


class Logger:
    def __init__(self):
        self.logfile = path_provider.get_csv('log.csv')
        # Initializes the log file
        if not os.path.exists(self.logfile):
            header = ['trial', 'observed', 'missed', 'waiting', 'planned', 'goal', 'time']
            with open(self.logfile, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                # write the header
                writer.writerow(header)

    def log(self, data: dict, verbose=True):
        """
        Logs some data, automatically numbering the trial.

        @param data: dictionary containing the keys 'observed_actions', 'time', 'prediction' and 'collab_len'
        @return: True if successful, False otherwise
        """
        with open(self.logfile, 'r+', encoding='UTF8') as f:
            # Obtains the last id used
            file_contents = csv.reader(f)
            n_logs = row_count = sum(1 for row in file_contents) - 1
            next_id = n_logs
            try:
                # Data insertion
                writer = csv.writer(f)
                row = [next_id, data['observed'], data['missed'], data['waiting'], data['planned'], data['goal'],
                       data['time']]
                writer.writerow(row)
                if verbose:
                    print("--- Log recorded ---")
            except KeyError:
                if verbose:
                    print("Failed to record log, missing data!")
                return False
        return True
