"""Utilities for opencv library"""
from __future__ import print_function
from __future__ import absolute_import

import cv2


class single_threaded_opencv(object):  # pylint: disable=invalid-name
    '''
    Context manager that disables IPP for deterministic results and sets number of threads to 0 to avoid
    hanging in multiprocessing forks
    '''
    def __init__(self):
        self._number_of_threads = None

    def __enter__(self):
        cv2.ipp.setUseIPP(flag=False)
        self._number_of_threads = cv2.getNumThreads()
        cv2.setNumThreads(0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cv2.setNumThreads(self._number_of_threads)
