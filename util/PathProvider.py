"""

Centralizes path retrieving.

"""

from pathlib import Path


class PathProvider:
    def __init__(self):
        self.BASEDIR = Path(__file__).parent.parent
        self.CONFIG = (self.BASEDIR / 'config').resolve()
        self.DATA = (self.BASEDIR / 'data').resolve()

    def get_domain(self, filename):
        return (self.DATA / 'CRADLE' / filename).resolve()

    def get_observations(self):
        return (self.DATA / 'CRADLE' / 'Observations.xml').resolve()

    def get_encodings(self):
        return (self.CONFIG / 'observation_encoding.csv').resolve()

    def get_save(self, filename):
        return (self.DATA / 'cognition' / filename).resolve()

    def get_pickle(self, filename):
        return (self.DATA / 'pickle' / filename).resolve()


path_provider = PathProvider()
