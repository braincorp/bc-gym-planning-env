""" Code for wrapping the motion primitive action in an object. """
from __future__ import division
from __future__ import absolute_import

import attr
import numpy as np


@attr.s
class Action(object):
    """ Object representing an 'action' - a motion primitive to execute in the environment """
    command = attr.ib(type=np.ndarray)

    def processed_command(self):
        """
        process the command, this is a temp solution to make the program compatible with both continuous and discrete
        action space.
        it returns [v, w] if input is (2,), discrete command index if input is (1,)
        """
        command_dict = {0: [0.2, 0.0],
                        1: [0.2, 0.3],
                        2: [0.2, -0.3],
                        3: [0.2, 0.5],
                        4: [0.2, -0.5],
                        5: [0.2, 0.7],
                        6: [0.2, -0.7],
                        7: [0.2, 1.1],
                        8: [0.2, -1.1],
                        9: [0.2, 1.3],
                        10: [0.2, -1.3]}
        if self.command.shape == (2,):
            pass
        elif self.command.shape == (1,):
            self.command = command_dict[self.command[0]]
        elif self.command.shape == (1, 1):
            self.command = np.array([0.2, self.command[0, 0]])
        else:
            raise NotImplementedError
