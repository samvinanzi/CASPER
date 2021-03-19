from controller import Supervisor, Robot

class Peppe:
    def __init__(self):
        self.supervisor = Supervisor()
        self.robot = self.supervisor.getFromDef("Peppe")
        self.timestep = int(self.supervisor.getBasicTimeStep())

    def move(self):
        trans_field = self.robot.getField("translation")
        while self.supervisor.step(self.timestep) != -1:
            # this is done repeatedly
            values = trans_field.getSFVec3f()
            print("MY_ROBOT is at position: %g %g %g" % (values[0], values[1], values[2]))


giuseppebaronedettopeppe = Peppe()
giuseppebaronedettopeppe.move()