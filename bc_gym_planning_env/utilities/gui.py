""" Gui for seeing what's inside the envs. """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import cv2
import numpy as np
import os
from sys import platform
from bc_gym_planning_env.envs.base.action import Action


class OpenCVGui(object):
    """Display an image with OpenCV"""
    def __init__(self):
        self._window_name = None

    def display(self, image):
        """
        Displau an image with Opencv. Prepare windows depending on the os
        :param image: numpy array with a BGR image
        :return np.array: numpy array with a BGR image
        """
        if self._window_name is None:
            self._window_name = "environment"
            cv2.namedWindow(self._window_name)
            cv2.moveWindow(self._window_name, 500, 200)
            if platform == "darwin":
                # bring window to front
                os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python" to true' ''')

        cv2.imshow(self._window_name, image)
        cv2.waitKey(1)
        return image

    def close(self):
        """Close possibly created window"""
        if self._window_name is not None:
            cv2.destroyWindow(self._window_name)


class KeyCapturePlay(object):
    """
    Used for playing the planning environment like a game.

    Its main goal is holding & updating state of sub-components:
    expert inputs, env and data recording in clear, independent steps.
    Example usage
    -------------
    p = HumanDataGatheringPlay(env, settings)
    p.pre_main_loop()
    while not p.done():
        p.before_env_step()
        p.env_step()
    """
    def __init__(self, env):
        """
        Initialize the wrapper of the environment.
        :param env object: the environment we want to wrap.
        """
        self._window_name = None
        self.env = env

        self._observation = None
        self._previous_observation = None
        self._img = None
        self._done = False
        self._action = None
        self._front_wheel_steering_rotation_state = np.float32(0)
        self._reward = np.float32(0)

    def _display(self, img):
        """
        Display an image with Opencv. Prepare windows depending on the os
        :param img: numpy array with a BGR image
        :return int: Returns int code of the key that has been detected as pressed by the window.
        """
        if self._window_name is None:
            self._window_name = "environment"
            cv2.namedWindow(self._window_name)
            cv2.moveWindow(self._window_name, 500, 200)
            if platform == "darwin":
                # bring window to front
                os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python" to true' ''')

        cv2.imshow(self._window_name, img)
        key = cv2.waitKey(0)
        return key

    def done(self):
        """
        Are we done?
        :return bool: are we done?
        """
        return self._done

    def pre_main_loop(self):
        """
        Perform initialization before the main data gathering loop.
        """
        self._observation = self._previous_observation = self.env.reset()
        self._img = self.env.render(mode='rgb_array')

    @staticmethod
    def _interpret(key):
        """
        Change a key press detected by opencv to action of the environment.
        :param key int: integer key code
        :return: base action associated with this ke
        """
        if key < 0:
            return None, None
        control_map = {
            ord('w'): (1, 0),
            ord('s'): (-1, 0),
            ord('a'): (0, 1),
            ord('d'): (0, -1)
        }
        try:
            return control_map[key]
        except KeyError:
            return None, None

    def before_env_step(self):
        """
        Things you need to do before the environment step:
        Pass inputs to agent, get back actions and metactions
        """
        # render the picture & wait for the user key input on it
        key = self._display(self._img)
        v, w = self._interpret(key)

        # interpret the key as action
        action = np.zeros(2, dtype=np.float32)
        if v is not None:
            action[0] = 0.5 * v
        if w is not None:
            coeff = np.pi / 2.0 * 0.1
            self._front_wheel_steering_rotation_state = np.clip(self._front_wheel_steering_rotation_state + coeff * w, -np.pi / 2.0, np.pi / 2.0)
            action[1] = self._front_wheel_steering_rotation_state

        print(80 * '=')
        print(action)

        self._action = Action(command=action)

    def env_step(self):
        """ Perform a step of the environment """
        self._previous_observation = self._observation
        self._observation, self._reward, done, _ = self.env.step(self._action)
        self._img = self.env.render(mode='rgb_array')
        self._done = done

    def post_env_step(self):
        """ Perform actions after the step of the environment """
        if len(self._observation.path):
            print("pose is {}, wanted pose is {}".format(self._observation.pose, self._observation.path[-1]))
        else:
            print("No path left!")

        if self._reward:
            print("reward = {}".format(self._reward))
