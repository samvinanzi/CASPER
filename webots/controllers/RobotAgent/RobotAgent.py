"""
RobotAgent controller.
Derives from Agent.
"""

from Agent import Agent


class RobotAgent(Agent):
    def __init__(self, name):
        Agent.__init__(self, supervisor=False)
        print(str(self.__class__.__name__) + " has activated.")

    def search_and_approach(self, objectname):
        """
        Tries to identify a target object and move towards it
        :param objectname: string, node name
        :return: True if successful, False otherwise
        """
        pass #todo
        # Observe for the target
        #objects = self.observe(True)
        #if objects is not None:
            #target = human.search_for_name(objectname, objects)
            #if target is not None:
                # If identified, gets its world position
            #    target_position = self.convert_to_2d_coords(target.getPosition())
            #    my_position = self.get_2d_position()
                # Moves from the current location to the target position
            #    self.walk([my_position, target_position], debug=True)
            #    return True
        #return False


# MAIN LOOP

robot = RobotAgent()

# Perform simulation steps until Webots is stopping the controller
while robot.step():
    pass
    #robot.calculate_blob_centroid()
    #objects = robot.observe(True)

    #robot.get_relative_coordinates(objects[0])
    #robot.show_camera_image(objects)
    #for object in objects:
    #    depth = robot.get_object_distance(object)
    #    print("Depth: " + str(depth))
    #pass

# Exit cleanup code.
