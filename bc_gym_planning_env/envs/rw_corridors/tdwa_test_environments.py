""" Helper code for making TDWA real data planning environment """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import requests
import shutil
import pickle
import sys

from bc_gym_planning_env.utilities.artifacts_utils import decompress_tar_archive, get_cache_key_path

from bc_gym_planning_env.utilities.costmap_2d import CostMap2D


def get_random_maps_squeeze_between_obstacle_in_corridor_on_path():
    """
    Download (or read from cache) randomly edited real world map with of a robot
    in a corridor with 3 square obstacles
    :return: a tuple of 3 elements
      - original realworld costmap of the corridor with 3 boxes
      - reference path that human took
      - tuple of 1000 randomized maps that were obstained by cutting and pasting 3 boxes randomly around
        their original localization
    """
    key = '413293a9-086b-46e0-83ef-bb09d23c70d5'
    test_maps_cache = 'test_random_maps_%s.pkl' % key

    session_tar_key = 'tdwa_paper_' + os.path.splitext(test_maps_cache)[0] + '.tar.xz'

    session_tar_file = get_cache_key_path(session_tar_key)
    tar_contents_folder = session_tar_file + '.contents/'
    if not os.path.exists(session_tar_file):
        r = requests.get(
            'https://s3-us-west-1.amazonaws.com/braincorp-research-public-datasets/' + session_tar_key
        )
        with open(session_tar_file, 'wb') as f:
            f.write(r.content)
        # remove unzipped cache if redownloaded
        if os.path.exists(tar_contents_folder):
            shutil.rmtree(tar_contents_folder)

    if not os.path.exists(tar_contents_folder):
        print('Unzipping Tar file "%s"' % session_tar_file)
        try:
            decompress_tar_archive(session_tar_file, tar_contents_folder, verbose=True)
        except Exception as ex:
            shutil.rmtree(tar_contents_folder)
            raise ex

    with open(os.path.join(tar_contents_folder, test_maps_cache), 'rb') as f:
        if sys.version_info > (3, 0):
            original_costmap = CostMap2D.from_state(pickle.load(f, encoding='latin1'))  # pylint: disable=unexpected-keyword-arg
            static_path = pickle.load(f, encoding='latin1')     # pylint: disable=unexpected-keyword-arg
            test_maps = tuple([CostMap2D.from_state(s) for s in pickle.load(f, encoding='latin1')])     # pylint: disable=unexpected-keyword-arg
        else:
            original_costmap = CostMap2D.from_state(pickle.load(f))
            static_path = pickle.load(f)
            test_maps = tuple([CostMap2D.from_state(s) for s in pickle.load(f)])

    return original_costmap, static_path, test_maps
