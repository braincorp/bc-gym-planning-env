from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import cv2
import os
from sys import platform


class OpenCVGui(object):
    """Display an image with OpenCV"""
    def __init__(self):
        self._window_name = None

    def display(self, image):
        """
        Displau an image with Opencv. Prepare windows depending on the os
        :param image: numpy array with a BGR image
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

    def close(self):
        """Close possibly created window"""
        if self._window_name is not None:
            cv2.destroyWindow(self._window_name)
