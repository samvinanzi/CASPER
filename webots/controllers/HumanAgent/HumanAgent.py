"""
HumanAgent controller.
Derives from Agent.
"""

from Agent import Agent
from controller import Node, Field
import math
import numpy as np
from path_planning.robot_astar import RoboAStar
from maps.Kitchen import Kitchen


class HumanAgent(Agent):
    def __init__(self, debug=False):
        Agent.__init__(self, debug=debug)

        self.object_in_hand: Node = None

        # Enables and disables devices
        self.camera.disable()
        #self.camera.recognitionEnable(self.timestep)
        self.rangefinder.disable()

        # Walking parameters
        self.BODY_PARTS_NUMBER = 13
        self.WALK_SEQUENCES_NUMBER = 8
        self.ROOT_HEIGHT = 1.27
        self.CYCLE_TO_DISTANCE_RATIO = 0.22
        self.speed = .5
        self.joints_position_field = []
        self.joint_names = [
            "leftArmAngle", "leftLowerArmAngle", "leftHandAngle",
            "rightArmAngle", "rightLowerArmAngle", "rightHandAngle",
            "leftLegAngle", "leftLowerLegAngle", "leftFootAngle",
            "rightLegAngle", "rightLowerLegAngle", "rightFootAngle",
            "headAngle"
        ]

        # Populates joints_position_field with the nodes
        root_node_ref = self.supervisor.getSelf()
        for i in range(0, self.BODY_PARTS_NUMBER):
            joint = root_node_ref.getField(self.joint_names[i])
            self.joints_position_field.append(joint)

        self.height_offsets = [
            # those coefficients are empirical coefficients which result in a realistic walking gait
            -0.02, 0.04, 0.08, -0.03, -0.02, 0.04, 0.08, -0.03
        ]
        self.angles = [  # those coefficients are empirical coefficients which result in a realistic walking gait
            [-0.52, -0.15, 0.58, 0.7, 0.52, 0.17, -0.36, -0.74],  # left arm
            [0.0, -0.16, -0.7, -0.38, -0.47, -0.3, -0.58, -0.21],  # left lower arm
            [0.12, 0.0, 0.12, 0.2, 0.0, -0.17, -0.25, 0.0],  # left hand
            [0.52, 0.17, -0.36, -0.74, -0.52, -0.15, 0.58, 0.7],  # right arm
            [-0.47, -0.3, -0.58, -0.21, 0.0, -0.16, -0.7, -0.38],  # right lower arm
            [0.0, -0.17, -0.25, 0.0, 0.12, 0.0, 0.12, 0.2],  # right hand
            [-0.55, -0.85, -1.14, -0.7, -0.56, 0.12, 0.24, 0.4],  # left leg
            [1.4, 1.58, 1.71, 0.49, 0.84, 0.0, 0.14, 0.26],  # left lower leg
            [0.07, 0.07, -0.07, -0.36, 0.0, 0.0, 0.32, -0.07],  # left foot
            [-0.56, 0.12, 0.24, 0.4, -0.55, -0.85, -1.14, -0.7],  # right leg
            [0.84, 0.0, 0.14, 0.26, 1.4, 1.58, 1.71, 0.49],  # right lower leg
            [0.0, 0.0, 0.42, -0.07, 0.07, 0.07, -0.07, -0.36],  # right foot
            [0.18, 0.09, 0.0, 0.09, 0.18, 0.09, 0.0, 0.09]  # head
        ]

        print(str(self.__class__.__name__) + " has activated.")

    def get_robot_position(self):
        """
        Using Supervisor functions, obtains the 2D (x,z) position of the robot.

        :return: (x,z) position of the robot
        """
        return self.convert_to_2d_coords(self.supervisor.getSelf().getPosition())

    def get_robot_orientation(self):
        """
        Using Supervisor functions, obtains the rotation matrix of the robot.

        :return: 3x3 rotation matrix
        """
        orientation = np.array(self.supervisor.getSelf().getOrientation())
        orientation.reshape(3, 3)
        return orientation

    def turn_towards(self, target):
        """
        Rotates the robot, in place, towards the target.

        :param target: coordinate in the format (x, z)
        :return: None
        """
        start = self.get_robot_position()
        end = target
        root_node_ref = self.supervisor.getSelf()
        root_rotation_field = root_node_ref.getField("rotation")
        # Rotate
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = math.atan2(dx, dy)
        while self.step():
            root_rotation_field.setSFRotation([0, 1, 0, angle])
            break

    def walk_simplified(self, target, speed=None, enable_bumper=False, debug=False):
        """
        Walks to a specific coordinate, without any animation.

        :param target: coordinate in the format (x, z)
        :param speed: walking speed in [m/s]
        :param enable_bumper: Activates or deactivates the bumper collision detection
        :param debug: activate/deactivate debug output
        :return: None
        """
        if speed is None:
            speed = self.speed
        # Calculates the waypoints (including the starting position)
        start = self.get_robot_position()
        end = target
        if debug:
            print("I am about to walk from {0} to {1}".format(start, end))
        # Retrieve the required nodes and fields
        root_node_ref = self.supervisor.getSelf()  # root_node_ref is the Robot node where the Supervisor is running
        root_translation_field = root_node_ref.getField("translation")
        root_rotation_field = root_node_ref.getField("rotation")

        # Rotate
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = math.atan2(dx, dy)
        if debug:
            print("Angle: {0} rad ({1}°)".format(angle, math.degrees(angle)))
            print("Distance: {0}".format(math.dist(start, end)))
        root_rotation_field.setSFRotation([0, 1, 0, angle])

        # Move
        s = np.array(start)
        e = np.array(end)
        # Line segment: f(0) = start, f(1) = end
        f = lambda t: s + t * (e - s)

        delta = 0.01 * speed    # When speed = 1, it ensures a movement of 1 m/s
        i = 0
        while self.step():
            if enable_bumper:
                collision = self.is_bumper_pressed()
            else:
                collision = False
            if not collision:
                # Steps forward and memorizes the new position
                i += delta
                waypoint = f(i)
            else:
                # Step backwards to avoid occupying the intersecting space (which will result in further detections)
                waypoint = f(i-delta)
            root_translation_field.setSFVec3f([waypoint[0], self.ROOT_HEIGHT, waypoint[1]])
            # If an object is held, it has to move together with the agent
            self.move_object_in_hand()
            # Stop if the walk is complete or if an obstacle was hit
            if i >= 1 or collision:
                if debug:
                    print("I have stopped in position: {0}".format(waypoint))
                return

    def walk(self, trajectory, speed=None, debug=False):
        # todo not currently working
        """
        Executes a walking animation to a specified target coordinate.

        :param trajectory: trajectory in the format [(x1, y1), (x2, y2), ...]
        :param speed: walking speed in [m/s]
        :return:
        """
        if speed is None:
            speed = self.speed
        # Calculates the waypoints (including the starting position)
        initial_position = self.get_robot_position()
        number_of_waypoints = len(trajectory) + 1  # Because I add the initial waypoint by hand
        waypoints = [[initial_position[0], initial_position[1]]]
        for i in range(0, number_of_waypoints - 1):
            waypoints.append([trajectory[i][0], trajectory[i][1]])
        final_waypoint = waypoints[-1]

        print("I am about to walk from {0} to {1}".format(initial_position, final_waypoint))

        root_node_ref = self.supervisor.getSelf()   # root_node_ref is the Robot node where the Supervisor is running
        root_translation_field = root_node_ref.getField("translation")
        root_rotation_field = root_node_ref.getField("rotation")
        #for i in range(0, self.BODY_PARTS_NUMBER):
        #    joint = root_node_ref.getField(self.joint_names[i])
        #    self.joints_position_field.append(joint)

        # compute waypoints distance
        waypoints_distance = []
        for i in range(0, number_of_waypoints):
            x = waypoints[i][0] - waypoints[(i + 1) % number_of_waypoints][0]
            z = waypoints[i][1] - waypoints[(i + 1) % number_of_waypoints][1]
            if i == 0:
                waypoints_distance.append(math.sqrt(x * x + z * z))
            else:
                waypoints_distance.append(waypoints_distance[i - 1] + math.sqrt(x * x + z * z))
        while self.step():
            time = self.supervisor.getTime()

            current_sequence = int(((time * speed) / self.CYCLE_TO_DISTANCE_RATIO) % self.WALK_SEQUENCES_NUMBER)
            # compute the ratio 'distance already covered between way-point(X) and way-point(X+1)'
            # / 'total distance between way-point(X) and way-point(X+1)'
            ratio = (time * speed) / self.CYCLE_TO_DISTANCE_RATIO - \
                    int(((time * speed) / self.CYCLE_TO_DISTANCE_RATIO))

            for i in range(0, self.BODY_PARTS_NUMBER):
                current_angle = self.angles[i][current_sequence] * (1 - ratio) + \
                                self.angles[i][(current_sequence + 1) % self.WALK_SEQUENCES_NUMBER] * ratio
                # Positions the joints
                self.joints_position_field[i].setSFFloat(current_angle)

            # adjust height
            current_height_offset = self.height_offsets[current_sequence] * (1 - ratio) + \
                                    self.height_offsets[(current_sequence + 1) % self.WALK_SEQUENCES_NUMBER] * ratio

            # move everything forward
            distance = time * speed
            relative_distance = distance - int(distance / waypoints_distance[number_of_waypoints - 1]) * \
                                waypoints_distance[number_of_waypoints - 1]

            for i in range(0, number_of_waypoints):
                if waypoints_distance[i] > relative_distance:
                    break

            distance_ratio = 0
            if i == 0:
                distance_ratio = relative_distance / waypoints_distance[0]
            else:
                distance_ratio = (relative_distance - waypoints_distance[i - 1]) / \
                                 (waypoints_distance[i] - waypoints_distance[i - 1])
            x = distance_ratio * waypoints[(i + 1) % number_of_waypoints][0] + (1 - distance_ratio) * waypoints[i][0]
            z = distance_ratio * waypoints[(i + 1) % number_of_waypoints][1] + (1 - distance_ratio) * waypoints[i][1]

            root_translation = [x, self.ROOT_HEIGHT + current_height_offset, z]
            angle = math.atan2(waypoints[(i + 1) % number_of_waypoints][0] - waypoints[i][0],
                               waypoints[(i + 1) % number_of_waypoints][1] - waypoints[i][1])
            rotation = [0, 1, 0, angle]
            root_translation_field.setSFVec3f(root_translation)
            root_rotation_field.setSFRotation(rotation)
            self.move_object_in_hand()

            # Check if destination is reached
            agent_position = human.get_gps_position()
            dist = math.hypot(agent_position[0] - final_waypoint[0], agent_position[1] - final_waypoint[1])
            if debug:
                print("Current position: " + str(agent_position))
                print("Target position: " + str(final_waypoint))
                print("Distance: " + str(dist))
                print("---")
            # Stop when at 40 cm from target
            if dist <= 0.4:
                # Let's reset the position to a neutral pose
                # Ensure feet are touching the ground
                root_translation_field.setSFVec3f([x, self.ROOT_HEIGHT, z])
                # Adopt a neutral position
                for i in range(0, self.BODY_PARTS_NUMBER):
                    self.joints_position_field[i].setSFFloat(0.0)
                # Compute an additional step to ensure that the position is reached
                #self.step()
                print("I have stopped at: " + str(self.get_robot_position()))
                return

    def approach_target(self, target_name, speed=None, debug=False):
        """
        Supervisor-version of search-and-approach: uses global knowledge of the world to identify the target coordinates
        without needing to visually identify the destination.

        :param target_name: str, name of the desired object
        :return: True if successful, False otherwise
        """
        assert isinstance(target_name, str), "Must specify a string name to search for"
        target = self.all_nodes.get(target_name)
        if target is not None:
            # If identified, gets its world position
            target_position = self.convert_to_2d_coords(target.getPosition())
            # Trace a path to the destination
            path = self.path_planning(target_position, show=False)
            if path is not None:
                print("Walking towards {0}".format(target_name))
                for waypoint in path:
                    self.walk_simplified(waypoint, speed=speed, debug=debug)
                self.turn_towards(target_position)
                return True
        return False

    def path_planning(self, goal, show=False):
        """
        Finds a path to reach a goal.

        :param goal: (x,z) coordinates.
        :param show: If True, visualizes the calculated path.
        :return: Path as a list of coordinates, or None if not found.
        """
        kitchen_map = Kitchen()     # todo This has to go outside of the HumanAgent class, higher up
        planner = RoboAStar(self.supervisor, kitchen_map, delta=0.45, min_distance=0.2, goal_radius=0.6)
        start = self.get_robot_position()
        if self.debug:
            print("[PATH-PLANNING] From {0} to {1}. Searching...".format(start, goal))
        path = list(planner.astar(start, goal, reversePath=False))
        if self.debug:
            print("[PATH-PLANNING] Path: {0}".format([waypoint for waypoint in path]))
        if path is None:
            print("[PATH-PLANNING] Unable to compute a path from {0} to {1}.".format(start, goal))
        elif show:
            for waypoint in path:
                kitchen_map.add_point_to_plot(waypoint)
            kitchen_map.visualize()
        return path

    def neutral_position(self):
        """
        Positions the human agent in a neutral position.

        :return: None
        """
        while self.step():
            for i in range(0, self.BODY_PARTS_NUMBER):
                self.joints_position_field[i].setSFFloat(0.0)
            break

    def hand_forward(self, steps=5):
        """
        Adopts a position where the hand is leaning frontally. Animated.

        :param steps: int, desired number of animation waypoints
        :return: None
        """
        assert isinstance(steps, int) and steps >= 1, "Steps must be an integer not lower than 1."
        # Start positions
        upper_start = 0
        lower_start = 0
        # Target angles, defined empirically
        upper_target = -0.485398
        lower_target = -0.3
        # Increments per step
        upper_step = (upper_target - upper_start) / steps
        lower_step = (lower_target - lower_start) / steps
        # Indexes in the joint field list
        upper_arm_index = self.joint_names.index("rightArmAngle")
        lower_arm_index = self.joint_names.index("rightLowerArmAngle")
        # Incremental positions
        upper_angle = upper_start
        lower_angle = lower_start
        while self.step():
            if math.fabs(upper_target - upper_angle) > 0 and math.fabs(lower_target - lower_angle) > 0:
                self.joints_position_field[upper_arm_index].setSFFloat(upper_angle)
                self.joints_position_field[lower_arm_index].setSFFloat(lower_angle)
                upper_angle += upper_step
                lower_angle += lower_step
            else:
                break
            #self.step()
            #break

    def grasp_object(self, target_name : str):
        """
        Positions the human in a grasping pose, then attaches the object (specified by its name) to the hand. Warning:
        the object is grasped even if it is not in range!

        :param target_name: str, name of the object to grasp
        :return: True if successful, False otherwise
        """
        assert isinstance(target_name, str), "Must specify a string name to search for"
        self.hand_forward()
        target: Node = self.all_nodes.get(target_name)
        if target is not None:
            self.object_in_hand = target
            #self.move_object_in_hand()
            self.object_in_hand.resetPhysics()
            hand_translation, hand_rotation = self.get_hand_transform()
            object_rotation = self.object_in_hand.getField("rotation").getSFRotation()
            # Maintain the object's angle
            hand_rotation[-1] = object_rotation[-1]
            while self.step():
                self.object_in_hand.getField("translation").setSFVec3f(hand_translation)
                self.object_in_hand.getField("rotation").setSFRotation(hand_rotation)
                break
            self.step()
            return True
        return False

    def release_object(self, destination):
        """
        Releases the hand-held object on a target destination (must be a surface)
        :param destination: str, name of the desired surface node
        :return: True if successful, False otherwise
        """
        assert isinstance(destination, str), "Must specify a string name to search for"
        if self.object_in_hand is not None:
            target: Node = self.all_nodes.get(destination)
            if target is not None:
                # Verifies that the destination is an acceptable surface
                accepted_types = ["Table"]
                if target.getTypeName() not in accepted_types:
                    print("Destination must be one of: {0}".format(accepted_types))
                else:
                    # The destination must have X and Z coordinates of the surface, Y equal to the table height
                    table_trans = target.getField("translation").getSFVec3f()
                    # Get the height of the table (Y)
                    table_size = target.getField("size").getSFVec3f()
                    table_height = table_size[1]
                    # Calculate the new position and assign it
                    final_position = [table_trans[0], table_height, table_trans[2]]

                    self.object_in_hand.getField("translation").setSFVec3f(final_position)
                    print("Deposited the {0} in {1}".format(self.object_in_hand.getTypeName(), final_position))
                    # Free the hand and reset the stance
                    self.object_in_hand = None
                    self.neutral_position()

                    return True
        else:
            print("Can't find destination {0} to release object!".format(destination))
        return False

    def move_object_in_hand(self):
        """
        Called to update the position of the held object to match the one of the hand.

        :return: None
        """
        if self.object_in_hand is not None:
            # Resets the object inertia, preventing it from being subject to gravitational effects
            self.object_in_hand.resetPhysics()
            hand_translation, hand_rotation = self.get_hand_transform()
            object_rotation = self.object_in_hand.getField("rotation").getSFRotation()
            # Maintain the object's angle
            hand_rotation[-1] = object_rotation[-1]
            self.object_in_hand.getField("translation").setSFVec3f(hand_translation)
            self.object_in_hand.getField("rotation").setSFRotation(hand_rotation)

    def get_hand_transform(self):
        """
        Obtains the translational and rotational vectors for the right hand.

        :return: hand translation, hand rotation wrt the world coordiantes
        """
        # Robot
        robot = self.supervisor.getSelf()
        robot_trans = robot.getField("translation").getSFVec3f()
        robot_rot = robot.getField("rotation").getSFRotation()
        robot_children = robot.getProtoField("children")
        robot_solid = robot_children.getMFNode(robot_children.getCount() - 1)
        robot_solid_children = robot_solid.getField("children")
        # Right upper arm
        upper_arm = robot_solid_children.getMFNode(3)
        upper_arm_endpoint = upper_arm.getField("endPoint")
        upper_arm_endpoint_solid = upper_arm_endpoint.getSFNode()
        upper_arm_trans = upper_arm_endpoint_solid.getField("translation").getSFVec3f()
        upper_arm_rot = upper_arm_endpoint_solid.getField("rotation").getSFRotation()
        upper_arm_children = upper_arm_endpoint_solid.getField("children")
        # Right lower arm
        hj = upper_arm_children.getMFNode(upper_arm_children.getCount() - 1)
        lower_arm_endpoint: Field = hj.getField("endPoint")
        lower_arm_endpoint_solid = lower_arm_endpoint.getSFNode()
        lower_arm_trans = lower_arm_endpoint_solid.getField("translation").getSFVec3f()
        lower_arm_rot = lower_arm_endpoint_solid.getField("rotation").getSFRotation()
        lower_arm_children = lower_arm_endpoint_solid.getField("children")
        # Hand
        hj2 = lower_arm_children.getMFNode(lower_arm_children.getCount() - 1)
        hand_endpoint: Field = hj2.getField("endPoint")
        hand_endpoint_solid = hand_endpoint.getSFNode()
        hand_children = hand_endpoint_solid.getField("children")
        # Hand_transform
        transform = hand_children.getMFNode(0)
        hand_trans = transform.getField("translation").getSFVec3f()
        hand_rot = transform.getField("rotation").getSFRotation()

        hand_position = transform.getPosition()
        # Calculates the translation from the hand to the world
        #translations = [robot_trans, upper_arm_trans, lower_arm_trans, hand_trans]
        rotations = [robot_rot, upper_arm_rot, lower_arm_rot, hand_rot] # disattivo?
        #t = [sum(x) for x in zip(*translations)]
        r = [sum(x) for x in zip(*rotations)]
        #return t, r
        return hand_position, r


# MAIN LOOP

human = HumanAgent()
speed = 2
debug = False

while human.step():
    """
    human.walk_simplified((-2.73, 4.05), speed=0.2)
    human.walk_simplified((-5.88, -4.21), speed=0.2)
    human.walk_simplified((4.38679, -4.8212), speed=0.2)
    human.walk_simplified((4.4642, -1.04344), speed=0.2)
    """
    if human.approach_target("coca-cola", speed, debug):
        human.grasp_object("coca-cola")
        if human.approach_target("table(1)", speed, debug):
            human.release_object("table(1)")
            human.walk_simplified((0, 0), speed, debug)
    break
