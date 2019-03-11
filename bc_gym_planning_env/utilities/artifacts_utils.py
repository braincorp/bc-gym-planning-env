""" Utilities for dealing with brain corp artifacts. """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import subprocess


def artifacts_cache_dir():
    """
    Standard folder for BrainCorp artifacts cache
    :return: path to standard folder for BrainCorp artifacts cache
    """
    base_folder = os.path.expanduser('~/.braincorp_artifacts')
    try:  # Try-except is safer than if exists, else due to possible race condition between test processes.
        os.makedirs(base_folder)
    except OSError:
        pass
    return base_folder


def get_cache_key_path(key):
    '''
    Return the full local path for the given key name.
    :param key: key to look up
    :return: path to the full local path for the given key name
    '''
    return os.path.join(artifacts_cache_dir(), key)


def decompress_tar_archive(tar_file, destination_folder, verbose=True):
    """
    Decompress folder with shell 'tar' command
    :param tar_file: path to a tar archive
    :param destination_folder: folder into which put unarchived content
    :param verbose: boolean flag whether to unarchive with verbose flag
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    if verbose:
        v_flag = 'v'
    else:
        v_flag = ''
    subprocess.call('tar -x%sf "%s" --directory "%s"' % (v_flag, tar_file, destination_folder), shell=True)
