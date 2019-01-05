from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np


def normalize_angle(z):
    '''
    Normalize angles to -pi to pi
    # http://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap
    '''
    return (np.array(z) + np.pi) % (2*np.pi) - np.pi


def inverse_transform(transform):
    """
    First unrotate, then untranslate

    Defined s.t T + inverse_transform(T) = (0, 0, 0)
    This means that the inverse flips the operator:  inverse_transform(T_i^j) = T_j^i.

    See the appendix (p. 328) in

    Tardos, J. D., Neira, J., Newman, P. M., & Leonard, J. J. (2002).
    Robust Mapping and Localization in Indoor Environments Using Sonar Data.
    The International Journal of Robotics Research, 21(4), 311-330.

    """
    c = np.cos(transform[..., 2])
    s = np.sin(transform[..., 2])

    x = -transform[..., 0] * c - transform[..., 1] * s
    y = transform[..., 0] * s - transform[..., 1] * c
    t = normalize_angle(-transform[..., 2])

    if transform.ndim == 2:
        return np.vstack((x, y, t)).T
    else:
        return np.array([x, y, t])


def homogenize(points):
    # type: (array(n_points,n_dim)[float]) -> array(n_points,n_dim+1)[float]
    """
    Add column of ones to NxM array to end up with Nx(M+1) array
    """
    return np.hstack((points, np.ones((points.shape[0], 1))))


def de_homogenize(points_h):
    # type: (array(n_points,n_dim+1)[float]) -> array(n_points,n_dim)[float]
    """
    Convert points from homogeneous coordinates into orinary coordinates.
    """
    return points_h[:, :-1] / points_h[:, [-1]].astype(np.float)


def rotation_matrix(rotation):
    """
    Construct 2D rotation matrix.

    :param: rotation: rotation [radians]
    :return: 2x2 rotation matrix.
    """
    c = np.cos(rotation)
    s = np.sin(rotation)
    return np.array([[c, -s],
                     [s, c]])


def transform_to_homogeneous_matrix(transform):
    """
    Represent transform as a 3x3 matrix that can be multiplied together to get a composite transform.
    Multiplying this on a vector of points augment with row of ones, project points between coordinate frames.

    If left-applied (i.e. as np.dot(M, point)), it first rotates and then translates
    :param transform: 3 numbers: 2 translation array [meters] and scalar rotation [radians]
    :return: 3x3 homogeneous transformation matrix
    """
    h = np.identity(3)
    h[:2, :2] = rotation_matrix(transform[2])
    h[:2, 2] = transform[:2]
    return h


def project_poses(transform, poses):
    """
    Python implementation of project_poses for the reference
    """
    if len(poses) == 0:
        return poses
    ph = homogenize(poses[:, :2])
    ph_moved = np.dot(transform_to_homogeneous_matrix(transform), ph.T).T
    ph_moved[:, 2] = normalize_angle(poses[:, 2] + transform[2])
    return ph_moved


def from_global_to_egocentric(global_poses, ego_pose_in_global_coordinates):
    transform = inverse_transform(np.asarray(ego_pose_in_global_coordinates))
    return project_poses(
        transform,
        global_poses
    )


def from_egocentric_to_global(ego_poses, ego_pose_in_global_coordinates):
    return project_poses(
        np.asarray(ego_pose_in_global_coordinates),
        ego_poses
    )
