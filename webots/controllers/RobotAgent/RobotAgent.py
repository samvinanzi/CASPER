"""
RobotAgent controller.
Derives from Agent.

TIAGo++: https://www.cyberbotics.com/doc/guide/tiagopp?version=master
"""
import math

from Agent import Agent
from controller import CameraRecognitionObject, Field, InertialUnit
import numpy as np


class RobotAgent(Agent):
    def __init__(self):
        Agent.__init__(self)

        self.all_nodes = self.obtain_all_nodes()
        self.motors = {'head_1': self.supervisor.getDevice("head_1_joint"),
                       'head_2': self.supervisor.getDevice("head_2_joint"),
                       'torso': self.supervisor.getDevice("torso_lift_joint"),
                       'arm_right_1': self.supervisor.getDevice("arm_right_1_joint"),
                       'arm_right_2': self.supervisor.getDevice("arm_right_2_joint"),
                       'arm_right_3': self.supervisor.getDevice("arm_right_3_joint"),
                       'arm_right_4': self.supervisor.getDevice("arm_right_4_joint"),
                       'arm_right_5': self.supervisor.getDevice("arm_right_5_joint"),
                       'arm_right_6': self.supervisor.getDevice("arm_right_6_joint"),
                       'arm_right_7': self.supervisor.getDevice("arm_right_7_joint"),
                       'arm_left_1': self.supervisor.getDevice("arm_left_1_joint"),
                       'arm_left_2': self.supervisor.getDevice("arm_left_2_joint"),
                       'arm_left_3': self.supervisor.getDevice("arm_left_3_joint"),
                       'arm_left_4': self.supervisor.getDevice("arm_left_4_joint"),
                       'arm_left_5': self.supervisor.getDevice("arm_left_5_joint"),
                       'arm_left_6': self.supervisor.getDevice("arm_left_6_joint"),
                       'arm_left_7': self.supervisor.getDevice("arm_left_7_joint"),
                       'wheel_left': self.supervisor.getDevice('wheel_left_joint'),
                       'wheel_right': self.supervisor.getDevice('wheel_right_joint')
                       }
        self.inertial = InertialUnit("inertial unit")
        self.inertial.enable(self.timestep)

        print(str(self.__class__.__name__) + " has activated.")

    def obtain_all_nodes(self):
        """
        Retrieves all the nodes from the current world and places them in a dictionary.

        :return: dictionary[name] = node
        """
        root_node = self.supervisor.getRoot()
        root_node_children: Field = root_node.getField("children")
        n = root_node_children.getCount()
        all_nodes = {}
        for i in range(n):
            node = root_node_children.getMFNode(i)
            name_field = node.getField("name")
            # We just care about objects, we discard non-named entities
            if name_field is not None:
                name = name_field.getSFString()
                all_nodes[name] = node
        return all_nodes

    def global_from_local(self, p_local):
        robot = self.supervisor.getSelf()
        R = np.array(robot.getOrientation())
        R = R.reshape(3, 3)
        tiago_rotation_correction = np.array([[-9.99961850e-01,  9.99999037e-01,  1.38730615e-03],
                                              [1.03062104e-04, -1.00138731e+00,  9.99999032e-01],
                                              [9.99999994e-01, -3.80068976e-05, -1.00010311e+00]])
        R -= tiago_rotation_correction
        T = robot.getPosition()
        tiago_translation_correction = np.array([0., 0., -0.24])
        p_local += tiago_translation_correction
        p_local[2] *= -1
        p_global = np.dot(R, p_local) + T
        return p_global

    def get_target_coordinates(self, target_name):
        assert isinstance(target_name, str), "Must specify a string name to search for"
        target = self.all_nodes.get(target_name)
        if target is not None:
            # If identified, gets its world position
            target_position = target.getPosition()
            print(target_position)
            return target_position

    def goto_position(self, positions: list, speed=1.0):
        """
        Moves the robot in a specified configuration.
        This function is usually not called directly, but accessed by another method instead.

        :param positions: list of 18 floats with the ordered joint positions.
        :return: None
        """
        assert len(positions) == 18, "List must have exactly 18 elements"
        assert isinstance(speed, float) and speed > 0.0, "Speed must be greater than 0"
        # Adopt the position
        while self.step():
            for i in range(len(positions)):
                motor = list(self.motors.values())[i]
                motor.setVelocity(motor.getMaxVelocity() / 1.0)
                motor.setPosition(positions[i])
            break

    def neutral_position(self):
        """
        Neutral position.

        :return: None
        """
        positions = [0.0, 0.0,                              # Head, torso
                     1.5, 1.4, -0.72, 1.5, -2.0, 1.39, 0,   # Left arm
                     1.5, 1.4, -0.72, 1.5, -2.0, 1.39, 0,   # Right arm
                     0.0, 0.0]                              # Wheels
        self.goto_position(positions)

    def rotate_90(self, direction, speed=0.5):
        """
        Rotates 90°, left or right.

        :param direction: 'left' or 'right'
        :param speed: float in (0.0, 1.0]
        :return: None
        """
        direction = direction.upper()
        assert direction == "LEFT" or direction == "RIGHT", "Direction must be 'left' or 'right'."
        assert 0.0 < speed <= 1.0, "Speed must be in (0, 1]."
        max_speed = 10.1523
        if direction == "LEFT":
            left_velocity = -speed * max_speed
            right_velocity = speed * max_speed
        else:
            left_velocity = speed * max_speed
            right_velocity = -speed * max_speed
        self.motors['wheel_left'].setPosition(float('inf'))
        self.motors['wheel_right'].setPosition(float('inf'))
        initial = self.inertial.getRollPitchYaw()
        while self.step():
            current = self.inertial.getRollPitchYaw()
            if math.fabs(initial[2] - current[2]) < 1.57:
                self.motors['wheel_left'].setVelocity(left_velocity)
                self.motors['wheel_right'].setVelocity(right_velocity)
            else:
                print("STOPPING")
                self.motors['wheel_left'].setVelocity(0.0)
                self.motors['wheel_right'].setVelocity(0.0)
                break

    def track_target(self, target_name, p_coefficient=0.1):
        while self.step():
            # Observe the environment for the specific target
            objects = self.observe()
            target = self.get_object_from_set(target_name, objects)
            if target is not None:
                # Calculate the distance from the center of the camera
                position_on_image = target.get_position_on_image()
                error = self.camera.getWidth() / 2 - position_on_image[0]
                current_position = self.motors['head_1'].getTargetPosition()    # Head motor position
                max_position = self.motors['head_1'].getMaxPosition()           # Head motor max rotation
                # Minimum distance of the target from the center of the camera to initialize movement
                deadzone = self.camera.getWidth() / 10
                if math.fabs(error) > deadzone:
                    diff = 0.02
                else:
                    diff = 0.0
                if error < 0:
                    diff *= -1
                new_position = current_position + diff
                if new_position > max_position:
                    if error > 0:
                        self.rotate_90("left")
                    else:
                        self.rotate_90("right")
                    new_position = 0
                self.motors['head_1'].setPosition(new_position)
            break



# MAIN LOOP

robot = RobotAgent()
tracked_objects = ['can', 'pedestrian']

# Perform simulation steps until Webots is stopping the controller
robot.neutral_position()
while robot.step():
    if robot.is_camera_active() and robot.is_range_finder_active():
        robot.track_target('pedestrian')
        #objects = robot.observe()
        #human: CameraRecognitionObject = robot.get_object_from_set('pedestrian', objects)
        #if human is not None:
        #    coordinates = robot.get_target_coordinates('human')
            #p_loc = human.get_position()
            #p_glob = robot.global_from_local(p_loc)
            #print("Local: {0}\nGlobal: {1}\n".format(p_loc, p_glob))

