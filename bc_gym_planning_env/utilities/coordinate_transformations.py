""" Utils for coordinate transformations. """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from builtins import zip

import numpy as np


try:
    from bc_gym_planning_env.utilities.numpy_utils import fast_hstack
except ImportError:
    fast_hstack = np.hstack


try:
    # import cpp optimized implementation if possible
    from brain.shining_utils.transform_utils import normalize_angle_impl
    normalize_angle = normalize_angle_impl
    '''
    Normalize angles to -pi to pi
    # http://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap
    :param angle: angle or angles to normalize
    :return: normalized angle or angles
    '''
except ImportError:
    def normalize_angle(z):
        """
        Normalize angles to -pi to pi
        # http://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap

        :param z: angles to normalize
        :return: normalized angles
        """
        return (np.array(z) + np.pi) % (2 * np.pi) - np.pi


try:
    from brain.shining_utils.transform_utils import inverse_transform_2d_impl
    inverse_transform = inverse_transform_2d_impl
    """
    First unrotate, then untranslate

    Defined s.t T + inverse_transform(T) = (0, 0, 0)
    This means that the inverse flips the operator:  inverse_transform(T_i^j) = T_j^i.

    See the appendix (p. 328) in

    Tardos, J. D., Neira, J., Newman, P. M., & Leonard, J. J. (2002).
    Robust Mapping and Localization in Indoor Environments Using Sonar Data.
    The International Journal of Robotics Research, 21(4), 311-330.
    :param transform Either[array(3)[float],  array(N, 3)[float]]: transform or transforms
    :return Either[array(3)[float],  array(N, 3)[float]]: inverted transform or transforms
    """
except ImportError:
    def inverse_transform(transform):
        """
        First unrotate, then untranslate

        Defined s.t T + inverse_transform(T) = (0, 0, 0)
        This means that the inverse flips the operator:  inverse_transform(T_i^j) = T_j^i.

        See the appendix (p. 328) in

        Tardos, J. D., Neira, J., Newman, P. M., & Leonard, J. J. (2002).
        Robust Mapping and Localization in Indoor Environments Using Sonar Data.
        The International Journal of Robotics Research, 21(4), 311-330.

        :param transform: transformation to invert
        :return: the inverted transformation

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


def rotation_matrix(rotation):
    """
    Construct 2D rotation matrix.

    :param rotation: rotation [radians]
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


def angle_diff(one, two):
    """ Compute the difference between angles (modulo 2pi)
    :param one float: the first angle
    :param two float: the second angle
    :return float: the difference between the angles
    """
    return np.mod(one - two, 2 * np.pi)


def cart2pol(x, y):
    """
    cartesian to polar coordinates on 2d point
    :param x: first dimension cartesian coordinate of the point
    :param y: second dimension cartesian coordinate of the point
    :return: the point in polar coordinates
    """
    """ cartesian to polar coordinates on 2d point """
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    phi = normalize_angle(phi)
    return r, phi


def pol2cart(r, phi):
    """
    polar to cartesian coordinates on 2d point
    :param r float: polar coordinate of the point - radius
    :param phi float: polar coordinate of the point - angle
    :return: the point in cartesian coordinates
    """
    return r * np.cos(phi), r * np.sin(phi)


def diff_angles(a1, a2):
    '''
    normalized to [-pi, pi) difference of angles: a1 - a2
    :param a1: first angle or angles
    :param a2: second angle or angles
    :return: normalized angle or angles
    '''
    return normalize_angle(a1 - a2)


def interpolate_angles(a1, a2):
    '''
    Returns angles that are in between of a1 and a2
    :param a1: first angle or angles
    :param a2: second angle or angles
    :return: angle or angles that are half-way in between
    '''
    diffs = diff_angles(a1, a2)
    return normalize_angle(a2 + 0.5 * diffs)


try:
    from brain.shining_utils.costmap_utils import world_to_pixel_impl
    world_to_pixel = world_to_pixel_impl
    """
    Convert a numpy set of world coordinates (... x 2 numpy array)
    to pixel coordinates, given origin ((x, y) in world coordinates)
    and resolution (in world units per pixel)

    The returned array is of type np.int, same shape as world_coords

    :param world_coords: An Array(..., 2)[float] array of (x, y) world coordinates in meters.
    :param origin: A (x, y) point representing the location of the origin in meters.
    :param resolution: Resolution in meters/pixel.
    :returns: An Array(..., 2)[int] of (x, y) pixel coordinates
    """
except ImportError:
    def world_to_pixel(world_coords, origin, resolution):
        """
        Convert a numpy set of world coordinates (... x 2 numpy array)
        to pixel coordinates, given origin ((x, y) in world coordinates)
        and resolution (in world units per pixel)

        The returned array is of type np.int, same shape as world_coords

        :param world_coords: An Array(..., 2)[float] array of (x, y) world coordinates in meters.
        :param origin: A (x, y) point representing the location of the origin in meters.
        :param resolution: Resolution in meters/pixel.
        :returns: An Array(..., 2)[int] of (x, y) pixel coordinates
        """
        assert len(origin) == 2

        assert isinstance(world_coords, np.ndarray)
        assert isinstance(origin, np.ndarray)

        assert world_coords.shape[world_coords.ndim - 1] == 2
        anti_resolution = 1./resolution
        return np.round((world_coords - origin) * anti_resolution).astype(np.int)


def world_to_voxel(world_coords, origin, xy_resolution, z_resolution):
    '''
    Convert a numpy set of world 3d coordinates (... x 3 numpy array)
    to voxel coordinates, given origin ((x, y, z) in world coordinates),
    xy_resolution and z_resolution (length of voxel sides in world units)

    The returned array is of type np.int, same shape as world_coords

    :param world_coords array(N, 3)[float]: world 3d coordinates
    :param origin array(3)[float]: origin of the volume (x, y, z) in world coordinates
    :param xy_resolution float: resolution of the costmap in meters per pixel.
    :param z_resolution Float: (length of voxel sides in world units)
    :return array(3)[int]: voxel coordinates in the volume
    '''
    world_coords = np.asarray(world_coords)
    assert world_coords.shape[world_coords.ndim - 1] == 3
    resolution = np.array([xy_resolution, xy_resolution, z_resolution])
    return np.round((world_coords - np.array(origin)) / resolution).astype(np.int)


def pixel_to_world(pixel_coords, origin, resolution):
    '''
    Convert a numpy set of pixel coordinates (... x 2 numpy array)
    to world coordinates, given origin ((x, y) in world coordinates) and
    resolution (in world units per pixel)

    The returned array is of type np.float32, same shape as pixel_coords
    :param pixel_coords array(N, 2)[float]: world (x, y) coordinates
    :param origin array(2)[float]: (x, y) coordinates of the left-bottom corner of the costmap in meters
    :param resolution float: resolution of the costmap in meters per pixel.
    :return array(2)[int]: pixel coordinates
    '''
    pixel_coords = np.asarray(pixel_coords)
    assert pixel_coords.shape[pixel_coords.ndim - 1] == 2
    return pixel_coords.astype(np.float32) * resolution + np.array(origin, dtype=np.float32)


def voxel_to_world(voxel_coords, origin, xy_resolution, z_resolution):
    '''
    Convert a numpy set of voxel coordinates (... x 3 numpy array)
    to world coordinates, given origin ((x, y, z) in world coordinates),
    xy_resolution and z_resolution (length of voxel sides in world units)
    :param voxel_coords array(N, 3)[int]: voxel coordinates
    :param origin array(3)[float]: origin of the volume (x, y, z) in world coordinates
    :param xy_resolution float: resolution of the costmap in meters per pixel.
    :param z_resolution Float: (length of voxel sides in world units)
    :return array(N, 3)[float32]: The returned array is of type np.float32, same shape as voxel_coords
    '''
    voxel_coords = np.asarray(voxel_coords)
    assert voxel_coords.shape[voxel_coords.ndim - 1] == 3
    resolution = np.array([xy_resolution, xy_resolution, z_resolution])
    return voxel_coords.astype(np.float32) * resolution + np.array(origin, dtype=np.float32)


def homogenize(points):
    """
    Add column of ones to NxM array to end up with Nx(M+1) array
    :param points array(N, 2)[float]: original points in euclidian
    :return array(N, 3)[float]: points in homogeneous coordinates
    """
    return fast_hstack((points, np.ones((points.shape[0], 1))))


def de_homogenize(points_h):
    """
    Convert points from homogeneous coordinates into orinary coordinates.
    :param points_h array(N, 3)[float]: points in homogeneous coordinates
    :return array(N, 2)[float]: points in euclidian coordinates
    """
    return points_h[:, :-1] / points_h[:, [-1]].astype(np.float)


def from_homogeneous_matrix(transform_matrix):
    '''
    Recover transform from transform_matrix
    :param transform_matrix array(3, 3)[float]: homogeneous transform matrix
    :return array(3)[float]: (x, y, angle) transform
    '''
    return np.array([transform_matrix[0, 2], transform_matrix[1, 2],
                     np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])])

try:
    from brain.shining_utils.env_utils import native_project_poses

    def project_poses(transform, poses):
        """
        Transform poses using the passed transform.
        The same as project_points but also rotates pose angles
        :param transform: pose transformation
        :param poses: poses to transform
        :return: transformed poses
        """
        if len(poses) == 0:
            return poses
        assert transform.shape[0] == 3
        assert poses.shape[1] == 3

        projected_poses_result = np.empty_like(poses)
        native_project_poses(transform, poses, projected_poses_result)
        return projected_poses_result

except ImportError:
    def project_poses(transform, poses):
        """
        Transform poses using the passed transform.

        Python implementation of project_poses for the reference
        :param transform: pose transformation
        :param poses: poses to transform
        :return: transformed poses
        """
        if len(poses) == 0:
            return poses
        if transform.shape[0] > 3:
            raise AssertionError("Too many elements for transform")
        if poses.shape[1] > 3:
            raise AssertionError("Too many elements for the poses")
        ph = homogenize(poses[:, :2])
        ph_moved = np.dot(transform_to_homogeneous_matrix(transform), ph.T).T
        ph_moved[:, 2] = normalize_angle(poses[:, 2] + transform[2])
        return ph_moved


def project_pose(transform, pose):
    """
    Non-batch version of project_poses
    :param transform: (x, y, angle) transformation
    :param pose: pose (x, y, angle) to transform
    :return: transformed pose
    """
    return project_poses(transform, np.array([pose]))[0]


def from_global_to_egocentric(global_poses, ego_pose_in_global_coordinates):
    """
    Project poses from global coordinates to egocentric coordinates

    :param global_poses: global poses
    :param ego_pose_in_global_coordinates: egocentric pose in global coordinates ==
                                          transofrmation  from egocentric pose to global pose
    :return: poses in egocentric coordinates
    """
    assert isinstance(ego_pose_in_global_coordinates, np.ndarray)
    transform = inverse_transform(ego_pose_in_global_coordinates)
    if global_poses.shape[1] == 3:
        return project_poses(
            transform,
            global_poses
        )
    else:
        assert global_poses.shape[1] == 2
        return project_points(
            transform,
            global_poses
        )


def from_egocentric_to_global(ego_poses, ego_pose_in_global_coordinates):
    """
    Project poses from global coordinates to egocentric poses

    :param ego_poses: egocentric poses
    :param ego_pose_in_global_coordinates: egocentric pose in global coordinates ==
                                          transofrmation  from egocentric pose to global pose
    :return: poses in global coordinates
    """
    assert isinstance(ego_pose_in_global_coordinates, np.ndarray)
    if ego_poses.shape[1] == 3:
        return project_poses(
            ego_pose_in_global_coordinates,
            ego_poses
        )
    else:
        assert ego_poses.shape[1] == 2
        return project_points(
            ego_pose_in_global_coordinates,
            ego_poses
        )


def project_points(transform, xy):
    """
    Use transform [x, y, angle] to project points from child coordinate frame to parent coordinate frame.

    If we have a transform, e.g. [1., 0., np.pi / 2.], we can interpret this as the translation of
    a frame in a parent frame (which would be at (0, 0, 0)). Lets call the parent frame W and the child frame
    C.

    We can *project* points from C to W by first rotating the points in the C reference frame (by np.pi / 2.) and
    then translating the points by (1.0, 0.)

    Example:
        project_points(
            robot_pose in global coordinates,
            points in egocentric frame of the robot_pose) == points in global coordinates

        project_points(
            transform between poseA and poseB,
            points in egocentric frame of poseB) == points in egocentric coordinates of poseA

    :param transform array(3)[float]: (x, y, angle) transform
    :param xy array(N, 2)[float]:  Nx2 points in child coordinate frame to project to parent coordinates
    :return array(N, 2)[float]: projected points
    """
    if len(xy) == 0:
        return xy
    ph = homogenize(xy)
    ph_moved = np.dot(transform_to_homogeneous_matrix(transform), ph.T).T
    return de_homogenize(ph_moved)


def from_rotation_matrix(rotation_matrix):
    """
    Recover angle from rotation matrix
    :param rotation_matrix array(2, 2)[float]: 2d rotation matrix
    :return float: angle from rotation matrix
    """
    return np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])


def rotation_matrix_t(rotation):
    """
    Construct 2D transposed 2D rotation matrix (i.e. same as rotation_matrix(rotation).T)

    :param rotation float: rotation [radians]
    :return array(2, 2)[float]: 2x2 rotation matrix.
    """
    return np.array([[np.cos(rotation), np.sin(rotation)], [-np.sin(rotation), np.cos(rotation)]])


def add_transforms(left, right):
    """
    O-plus (+) composition operator. First translates, then rotates.

    Note that this is *not* a commutative operation, but a shorthand for matrix multiplication.

    right.x is forward translation (w.r.t *robot*).

    Equivalent to matrix multiplication of left and right

    Example:
        intermediate_pose = transform_pose(pose, left)
        result_pose = transform_pose(intermediate_pose, right)
        equivalent to
        result_pose = transform_pose(pose, add_transforms(left, right))

    :param left array(3)[float]: First transform to add [x, y, angle]
    :param right array(3)[float]: Second transform to add [x, y, angle]
    :return array(3)[float]: new transform C = self (+) right
    """
    assert left.ndim == 1
    c = np.cos(left[2])
    s = np.sin(left[2])
    x = left[0] + c * right[0] - s * right[1]
    y = left[1] + s * right[0] + c * right[1]
    t = normalize_angle(left[2] + right[2])
    return np.array([x, y, t])


def chain_transforms(*transforms):
    """
    Helper to add transform in a collection using add_transforms
    :param transforms Collection[array(3)[float64]]: collection of 2d transforms
    :return array(3)[float64]: final transform
    """
    assert len(transforms) > 0
    t = transforms[0]
    for new_t in transforms[1:]:
        t = add_transforms(t, new_t)
    return t


def add_transforms_batch(left, right):
    '''
    Efficient way to add multiple transform arrays at once
    :param left: array of transforms
    :param right: array of transforms
    :return: array pairwise additions of input transforms
    '''
    assert left.ndim == 2
    c = np.cos(left[:, 2])
    s = np.sin(left[:, 2])
    x = left[:, 0] + c * right[:, 0] - s * right[:, 1]
    y = left[:, 1] + s * right[:, 0] + c * right[:, 1]
    t = normalize_angle(left[:, 2] + right[:, 2])
    return np.vstack((x, y, t)).T


def transform_pose(pose, *transforms):
    '''
    Move pose according to transforms (translation then rotation)
    Example:
        transform_pose(robot_pose_in_global_frame, odom_transform) == robot pose in global after movement
    :param pose array(3)[float64]: pose [x, y, angle]
    :param transforms Collection[array(3)[float64]]: one or more transforms [x, y, angle]
    :return array(3)[float64]: new pose
    '''
    for t in transforms:
        pose = add_transforms(pose, t)
    return pose


def transform_poses_batch(poses, transforms):
    """
    Transform multiple poses with multiple transforms at once
    :param poses array(N, 3)[float64]: array of poses
    :param transforms array(N, 3)[float64]: array of transforms
    :return array(N, 3)[float64]: transformed poses
    """
    assert(transforms.ndim > 1 or poses.ndim > 1)
    if transforms.ndim == 1:
        transforms = np.tile(transforms, (len(poses), 1))

    if poses.ndim == 1:
        poses = np.tile(poses, (len(transforms), 1))

    assert transforms.shape == poses.shape

    return add_transforms_batch(poses, transforms)


def transforms_between_poses(poses):
    '''
    Given chain of poses, extract transforms between them
    Examples:
        - if poses are measured by an odometry system, this function returns a chain of odometry transformations
        that if applied sequentially, move the robot along the path.

        - if poses = [pose_A in global coordinates, pose_B in global coordinates],
        then transforms_between_poses(poses)[0] is pose_B as viewed in a egocentric coordinate frame of pose_A
    :param poses array(N, 3)[float64]: chain of poses
    :return array(N-1, 3)[float64]: transforms between subsequent poses
    '''
    return transforms_between_poses_implementation(
        np.diff(poses, axis=0),
        poses[:-1, 2]
    )


def transforms_between_poses_implementation(poses_diffs, initial_angles):
    '''
    A helper for computing transforms between pose diffs
    We need to rotate the translation in the odometry frame to a translation in the pose_a frame;
    The rotation delta can be obtained directly from the poses since we are in the plane.
    Below is a batch version, equivalent to the per-element version
    r = rotation_matrix(-robot_poses[i, 2])
    delta[i] = np.dot(r, np.array([dx[i], dy[i]]))
    :param poses_diffs array(N, 3)[float64]: array of pose differences
    :param initial_angles array(N)[float64]: initial angles
    :return array(N, 3)[float64]: corresponding transforms
    '''
    poses_diffs[:, 2] = normalize_angle(np.ascontiguousarray(poses_diffs[:, 2]))

    poses_cos = np.cos(initial_angles)
    poses_sin = np.sin(initial_angles)
    dx, dy, da = poses_diffs[:, 0], poses_diffs[:, 1], poses_diffs[:, 2]
    return np.array([
        poses_cos * dx + poses_sin * dy,
        -poses_sin * dx + poses_cos * dy,
        da
    ]).T


def transform_between_poses(src_pose, dst_pose):
    """
    Non-batch transforms_between_poses convenience function
    :param src_pose array(3)[float]: first pose (x, y, angle)
    :param dst_pose array(3)[float]: second pose (x, y, angle)
    :return array(3)[float]: transform between poses
    """
    return transforms_between_poses(np.array([src_pose, dst_pose]))[0]


def transforms_between_pose_pairs(src_poses, dst_poses):
    '''
    Computes vector of transforms between src_poses to dst poses:
    src_pose0-t0-dst_pose0
    src_pose1-t1-dst_pose1
    ...

    Note that it is a different from transforms_between_poses which is a chain version of transforms between poses:
    pose0-t0-pose1-t1-pose2 ..
    :param src_poses array(N, 3)[float]: first set of poses (x, y, angle)
    :param dst_poses array(N, 3)[float]: second set of pose (x, y, angle)
    :return array(N-1, 3)[float]: transform between poses
    '''
    return transforms_between_poses_implementation(
        dst_poses - src_poses,
        src_poses[:, 2]
    )


try:
    from brain.shining_utils.env_utils import native_compose_covariances

    def compose_covariances(t_a_b, s_a_b, t_b_c, s_b_c, debug=False):
        """
        For [x, y, angle] transform t_a^b, t_b^c s.t. t_a^b + t_b^c = t_a^c

        and s_a_b the covariance of transformation B to A, J_A the Jacobian of the parameters of C w.r.t. the parameters
        of A (and analogously for B). The Covariance of C is then:

        Sigma_c = J_A s_a_b J_A^T + J_B s_b_c J_B^T eqn. (2.6) in Olson (2008)
        [http://rvsn.csail.mit.edu/static-content/eolson/eolson_phd_thesis_2008.pdf]

        :param t_a_b: angle of transform B to A (e.g. T_a^b)
        :param s_a_b: Covariance matrix for Transform A
        :param t_b_c: x and y of transform C to B (e.g. T_b^c)
        :param s_b_c: Covariance matrix for Transform B
        :param debug: run strict check on covariance matrix

        :return: the composed covariance matrix associated with T_a^c
        """
        # pylint: disable=invalid-name
        t_a_b_theta = float(t_a_b[2])
        tbx = float(t_b_c[0])
        tby = float(t_b_c[1])

        jacobian_a = np.zeros((3, 3), dtype=np.float64)
        jacobian_b = np.zeros((3, 3), dtype=np.float64)

        native_compose_covariances(t_a_b_theta, tbx, tby, jacobian_a, jacobian_b)

        # Eqn. (2.6)
        covariance = np.dot(np.dot(jacobian_a, s_a_b), jacobian_a.T) + np.dot(np.dot(jacobian_b, s_b_c), jacobian_b.T)

        if debug:
            # Sanity check on composed covariance matrix
            check_valid_cov_mat(covariance)

        return covariance
except ImportError:

    def compose_covariances(t_a_b, s_a_b, t_b_c, s_b_c, debug=False):
        """
        For [x, y, angle] transform t_a^b, t_b^c s.t. t_a^b + t_b^c = t_a^c

        and s_a_b the covariance of transformation B to A, J_A the Jacobian of the parameters of C w.r.t. the parameters
        of A (and analogously for B). The Covariance of C is then:

        Sigma_c = J_A s_a_b J_A^T + J_B s_b_c J_B^T eqn. (2.6) in Olson (2008)
        [http://rvsn.csail.mit.edu/static-content/eolson/eolson_phd_thesis_2008.pdf]

        :param t_a_b: angle of transform B to A (e.g. T_a^b)
        :param s_a_b: Covariance matrix for Transform A
        :param t_b_c: x and y of transform C to B (e.g. T_b^c)
        :param s_b_c: Covariance matrix for Transform B
        :param debug: run strict check on covariance matrix

        :return: the composed covariance matrix associated with T_a^c
        """
        jacobian_a = np.identity(3)
        jacobian_a[0, 2] = -np.sin(t_a_b[2]) * t_b_c[0] - np.cos(t_a_b[2]) * t_b_c[1]
        jacobian_a[1, 2] = np.cos(t_a_b[2]) * t_b_c[0] - np.sin(t_a_b[2]) * t_b_c[1]

        jacobian_b = np.identity(3)
        jacobian_b[0, 0] = np.cos(t_a_b[2])
        jacobian_b[0, 1] = -np.sin(t_a_b[2])
        jacobian_b[1, 0] = np.sin(t_a_b[2])
        jacobian_b[1, 1] = np.cos(t_a_b[2])

        # Eqn. (2.6)
        covariance = np.dot(np.dot(jacobian_a, s_a_b), jacobian_a.T) + np.dot(np.dot(jacobian_b, s_b_c), jacobian_b.T)

        if debug:
            # Sanity check on composed covariance matrix
            check_valid_cov_mat(covariance)

        return covariance


def compose_covariances_batch(ts_a_b, ss_a_b, ts_b_c, ss_b_c):
    '''
    Batch version of compose_covariances. Transfroms and covariances are arrays of the same length
    :param ts_a_b: angles of transform B to A (e.g. T_a^b)
    :param ss_a_b: Covariance matrices for Transform A
    :param ts_b_c: angles transform C to B (e.g. T_b^c)
    :param ss_b_c: Covariance matrices for Transform B
    :return: array of covariances
    '''
    result = []
    for t_a_b, s_a_b, t_b_c, s_b_c in zip(ts_a_b, ss_a_b, ts_b_c, ss_b_c):
        result.append(compose_covariances(t_a_b, s_a_b, t_b_c, s_b_c))
    return np.array(result)


def add_uncertain_transforms(transform0_covariance0, transform1_covariance1):
    """
    Wrapper to add uncertain transforms (transform + covariance)
    :param transform0_covariance0 (array(3)[float64], array(3, 3)[float64]): first transform and covariance pair
    :param transform1_covariance1 (array(3)[float64], array(3, 3)[float64]): second transform and covariance pair
    :return (array(3)[float64], array(3, 3)[float64]): added transforms and covariances
    """
    transform0, covariance0 = transform0_covariance0
    transform1, covariance1 = transform1_covariance1
    return add_transforms(transform0, transform1), compose_covariances(transform0, covariance0, transform1, covariance1)


def check_valid_cov_mat(cov_mat):
    """
    Strict check that covariance matrix is well-formed
    :param cov_mat: 3x3 covariance matrix
    """
    assert np.all(np.sign(np.diag(cov_mat)) >= 0), "Variances must be >= 0. Something is fishy. {}".format(
        str(np.diag(cov_mat)))
    np.testing.assert_array_almost_equal(cov_mat, cov_mat.T)
    assert (np.linalg.det(cov_mat[0:2, 0:2]) >= 0 and np.linalg.det(cov_mat) >= 0), \
        "covariance matrix is not positive semi-definite"


def normalize_transform_angle(transform):
    '''
    Call normalize_angle on the angle of a transform and return a normalized copy
    :param transform array(3)[float64]: (x, y, angle) transform
    :return array(3)[float64]: transform with normalized angle
    '''
    result = transform.copy()
    if result.ndim == 1:
        result[2] = normalize_angle(float(result[2]))
    else:
        result[..., 2] = normalize_angle(np.ascontiguousarray(result[..., 2]))
    return result


def cumsum_uncertain_transforms(transforms, covariances):
    '''
    The same as np.cumsum, but instead of numbers are pairs of transforms and covariances
    and '+' is add_uncertain_transforms
    :param transforms: np.array of transforms
    :param covariances: np.array of covariances
    :return: (cumsum of transforms, cumsum of covariances)
    '''
    result_transforms = [transforms[0]]
    result_covariances = [covariances[0]]
    t = (transforms[0], covariances[0])
    for next_t in zip(transforms[1:], covariances[1:]):
        t = add_uncertain_transforms(t, next_t)
        result_transforms.append(t[0])
        result_covariances.append(t[1])
    return np.array(result_transforms), np.array(result_covariances)


def cumsum_transforms(transforms):
    '''
    The same as np.cumsum, but instead of numbers are transforms
    and '+' is add_transforms
    :param transforms: np.array of transforms
    :return: cumsum of transforms
    '''
    result_transforms = [transforms[0]]
    t = transforms[0]
    for next_t in transforms[1:]:
        t = add_transforms(t, next_t)
        result_transforms.append(t)
    return np.array(result_transforms)


def rigid_body_pose_transform(transform_in_child_frame, child_to_parent_transform):
    '''
    How is pose on a rigid body (pose_on_rigid_body) transformed if the whole
    body is transformed with rigid_body_transform.
    Example:
        pose_on_rigid_body == baselink to lidar
        rigid_body_transform == odom transform of the baselink between frames
        rigid_body_pose_transform(rigid_body_transform, baselink to lidar) == how lidar pose is transformed between frames

    Equivalently, how would transform between elements (rigid_body_transform) change
    if elements were first transformed by another transform (pose_on_rigid_body)?
    Example:
        r = some transform
        t = transform_between_poses(pose0, pose1)
        rigid_body_pose_transform(t, r) == transform_between_poses(transform_pose(pose0, r), transform_pose(pose1, r))

    Equivalently, how would transform (rigid_body_transform) between elements change if elements were in a different basis
    where basis transform is pose_on_rigid_body.

    :param transform_in_child_frame: transform in a parent coordinate frame
    :param child_to_parent_transform: transform from parent to child
    :return: transform between points in a parent coordinate frame.
    '''
    return add_transforms(add_transforms(
        inverse_transform(child_to_parent_transform),
        transform_in_child_frame),
        child_to_parent_transform)


def rigid_body_pose_transforms_batch(transforms_in_child_frame, child_to_parent_transforms):
    '''
    Batch version of rigid_body_pose_transform
    :param transforms_in_child_frame: transform in a child coordinate frame
    :param child_to_parent_transforms: transform from child to parent
    :return: transform between points in a parent coordinate frame.
    '''
    if child_to_parent_transforms.ndim == 1:
        child_to_parent_transforms = np.tile(child_to_parent_transforms, (len(transforms_in_child_frame), 1))
    else:
        assert child_to_parent_transforms.shape == transforms_in_child_frame.shape
    return add_transforms_batch(add_transforms_batch(
        inverse_transform(child_to_parent_transforms),
        transforms_in_child_frame),
        child_to_parent_transforms)


def transformation_between_coordinate_frames(pose_in_parent_frame, pose_in_child_frame):
    '''
    If you have coordinates of the same pose in two different coordinate frames (parent and child)
    then you can recover a transform T, such that
    project_pose(T, pose1) == pose0.
    T = add_transforms(pose1, inverse_transform(pose0))
    T projects poses from child to parent frame
    :param pose_in_parent_frame array(3)[float]: pose (x, y, ange) in the parent coordinate frame
    :param pose_in_child_frame array(3)[float]: pose (x, y, ange) in the child coordinate frame
    :return: transform from coordinate frame 0 to frame 1
    '''
    return add_transforms(pose_in_parent_frame, inverse_transform(pose_in_child_frame))


def get_map_to_odom_tranformation(map_pose, odom_pose):
    """
    Common transformation utility function for map and odom frames: returns transform between two poses
    :param map_pose array(3)[float64]: pose in map frame
    :param odom_pose array(3)[float64]: pose in odom frame
    :return array(3)[float64]: transform between poses
    """
    return transformation_between_coordinate_frames(pose_in_parent_frame=map_pose, pose_in_child_frame=odom_pose)


def from_map_to_odom(map_poses, map_to_odom_transformation):
    """
    Common transformation utility function for map and odom frames: transform poses from map to odom frame
    :param map_poses array(..., 3)[float64]: poses in map frame
    :param map_to_odom_transformation array(3)[float64]: transformation from map to odom
    :return array(..., 3)[float64]: odom poses
    """
    if map_poses.ndim == 1:
        return from_global_to_egocentric(np.ascontiguousarray(map_poses[None, :]), map_to_odom_transformation)[0]
    else:
        return from_global_to_egocentric(map_poses, map_to_odom_transformation)


def from_odom_to_map(odom_poses, map_to_odom_transformation):
    """
    Common transformation utility function for map and odom frames: transform poses from odom to map frame
    :param odom_poses array(..., 3)[float64]: odom in map frame
    :param map_to_odom_transformation array(3)[float64]: transformation from map to odom
    :return array(..., 3)[float64]: map poses
    """
    if odom_poses.ndim == 1:
        return from_egocentric_to_global(odom_poses.reshape(1, 3), map_to_odom_transformation)[0]
    else:
        return from_egocentric_to_global(odom_poses, map_to_odom_transformation)


def apply_3d_transform_matrix(
        transform_matrix,  # type: ndarray(4, 4)[float]
        points,  # type: ndarray(N, 3)[float]
    ):  # -> type: ndarray(N, 3)[float]
    """
    Transform a set of points given a 4x4 homogeneous transform matrix.
    :param transform_matrix: A homogeneous transform matrix
    :param points: An array of N points in 3d
    :return: A new array of points after being transformed by the matrix.
    """
    assert points.shape[1] == 3
    extended = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed = np.dot(transform_matrix, extended.T).T
    return transformed[:, :3]
