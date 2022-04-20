"""

Centralizes path retrieving.

"""

from pathlib import Path


class PathProvider:
    def __init__(self):
        self.BASEDIR = Path(__file__).parent.parent
        self.CONFIG = (self.BASEDIR / 'config').resolve()
        self.DATA = (self.BASEDIR / 'data').resolve()
        self.IMG = (self.BASEDIR / 'img').resolve()
        self.MOTION = (self.BASEDIR / 'webots' / 'controllers' / 'RobotAgent' / 'motions').resolve()
        self.ONTO = (self.BASEDIR / 'ontology').resolve()

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

    def get_csv(self, filename):
        return (self.DATA / 'csv' / filename).resolve()

    def get_image(self, filename):
        return (self.IMG / filename).resolve()

    def get_robot_motion(self, filename):
        return (self.MOTION / filename).with_suffix('.motion').resolve()

    def get_ontology(self, filename):
        return str((self.ONTO / filename).with_suffix('.owl').resolve())    # Owlready can't handle PosixPath apparently

    def get_SQLite3(self):
        return (self.DATA / 'sqlite3' / 'kitchen.sqlite3').resolve()



path_provider = PathProvider()
