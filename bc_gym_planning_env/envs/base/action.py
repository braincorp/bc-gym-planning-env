""" Code for wrapping the motion primitive action in an object. """
from __future__ import division
from __future__ import absolute_import

import numpy as np


class Action(object):
    """ Object representing an 'action' - a motion primitive to execute in the environment """

    def __init__(self, command=None):
        """
        initialization
        :param command np.ndarray: [v, w] if input is (2,), discrete command index if input is (1,)
        """
        self._command_dict = {0: [0.2, 0.0],
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
        if command.shape == (2,):
            self.command = command
        elif command.shape == (1,):
            #self.command = self._command_dict[command[0]]
            self.command = np.array([0.2, command[0]])
        elif command.shape == ():
            self.command = self._command_dict[int(command)]
        elif command.shape == (1,1):
            self.command = np.array([0.2, command[0, 0]])
        else:
            raise NotImplementedError
