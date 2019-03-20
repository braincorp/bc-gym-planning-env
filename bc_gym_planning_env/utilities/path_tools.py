""" Toolset for the path. """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from builtins import range
import numpy as np

import cv2

from bc_gym_planning_env.utilities.coordinate_transformations import normalize_angle, diff_angles
from bc_gym_planning_env.utilities.numpy_utils import fast_amin, fast_amax


def get_blit_mask(patch, im, x, y):
    '''
    Helper for blit operations. Computes a blit mask given a patch
    :param patch: patch A 2D numeric value mask with positive numbers being evaluated as true
    :param im: The 2/3D image to blit
    :param x: The X location on the image to place  the center of the patch
    :param y: The Y location on the image to place the center of the patch
    :return: A subset of the input image that the patch's bounding box encompasses. Note that this image may not
        necessarily be the same size as the patch. The second tuple is the patch being returned as a binary mask.
    '''
    patch_h, patch_w = patch.shape[:2]
    im_h, im_w = im.shape[:2]

    # We compute the end points for a bounding box that holds the patch with
    # its center point at x, y.
    #
    start_x, start_y = int(x - patch_w // 2), int(y - patch_h // 2)
    end_x, end_y = start_x + patch_w, start_y + patch_h

    # If the starting point is larger than the end of the image, or the
    # ending point of the bounding box is smaller than the beginning of
    # the image then we know we have no overlap and can return immediately
    if start_x >= im_w or start_y >= im_h or end_x <= 0 or end_y <= 0:
        return None, None

    # We have to account for cases where the patch is bigger than the image,
    # as well as for cases where some portion of the bounding box falls
    # outside of the image.
    left_x_offset, right_x_offset = max(0, -start_x), max(0, end_x - im_w)
    top_y_offset, bottom_y_offset = max(0, -start_y), max(0, end_y - im_h)

    # This is the portion of the image that corresponds to the bounding box produced
    # by the patch.
    if im.ndim == 3:
        image_slice = im[start_y + top_y_offset:end_y - bottom_y_offset, start_x + left_x_offset:end_x - right_x_offset, ...]
    else:
        # using ... is slow for normal 2d costmap
        image_slice = im[start_y + top_y_offset:end_y - bottom_y_offset, start_x + left_x_offset:end_x - right_x_offset]

    # We return the region of interest and also convert the patch to a boolean mask
    return image_slice, patch[top_y_offset:patch_h - bottom_y_offset, left_x_offset:patch_w - right_x_offset] > 0


def blit(patch, im, x, y, color, axis=None, alpha = 1.0):
    '''
    superimpose a binary patch, or the part of it that is within bounds,
    over image im, centered at (x, y), with given color
    :param patch: patch A 2D numeric value mask with positive numbers being evaluated as true
    :param im: The 2/3D image to blit
    :param x: The X location on the image to place  the center of the patch
    :param y: The Y location on the image to place the center of the patch
    :param color: 3-tuple representing the color
    :param axis: tuple of color indices - which color axis to mark with 'color' (None - mark all)
    :param alpha: transparency of the color
    '''
    image_slice, blit_mask = get_blit_mask(patch, im, x, y)
    if image_slice is None:
        return

    if axis is None:
        image_slice[blit_mask, ...] = (image_slice[blit_mask, ...]*(1-alpha) + np.array(color) * alpha) if alpha < 1.0 else np.array(color)
    else:
        image_slice[blit_mask, axis[0]:axis[1]] = image_slice[blit_mask, axis[0]:axis[1]]*(1-alpha) + np.array(color) * alpha if alpha < 1.0 else np.array(color)


def get_blit_values(patch, im, x, y):
    """
    like blit, but just returns the values of the im at the
    pixels that would be set by blit.

    :param patch: patch A 2D numeric value mask with positive numbers being evaluated as true
    :param im: The 2/3D image to blit
    :param x: The X location on the image to place  the center of the patch
    :param y: The Y location on the image to place the center of the patch
    :return: returns the values of the im at the pixels that would be set by blit
    """
    image_slice, blit_mask = get_blit_mask(patch, im, x, y)
    if image_slice is None:
        return np.empty((0,) + im.shape[2:], dtype=im.dtype)
    if im.ndim == 3:
        return image_slice[blit_mask, ...]
    else:
        # using ... is slow for normal 2d costmap
        return image_slice[blit_mask]


try:
    from brain.shining_utils.costmap_utils import get_pixel_footprint_impl
    get_pixel_footprint = get_pixel_footprint_impl
    '''
    Return a binary image of a given robot footprint, in pixel coordinates,
    rotated over the appropriate angle range.
    Point (0, 0) in world coordinates is in the center of the image.
    angle_range: if a 2-tuple, the robot footprint will be rotated over this range;
        the returned footprint results from superimposing the footprint at each angle.
        If a single number, a single footprint at that angle will be returned
    robot_footprint: n x 2 numpy array with ROS-style footprint (x, y coordinates),
        in metric units, oriented at 0 angle
    map_resolution: 
    :param angle Float: orientation of the robot 
    :param robot_footprint array(N, 2)[float64]: n x 2 numpy array with ROS-style footprint (x, y coordinates),
        in metric units, oriented at 0 angle
    :param map_resolution Float: length in metric units of the side of a pixel 
    :param fill bool: if True, the footprint will be solid; if False, only the contour will be traced
    :return array(K, M)[uint8]: image of the footprint drawn on the image in white 
    '''
except ImportError:
    def get_pixel_footprint(angle, robot_footprint, map_resolution, fill=True):
        '''
        Return a binary image of a given robot footprint, in pixel coordinates,
        rotated over the appropriate angle range.
        Point (0, 0) in world coordinates is in the center of the image.
        angle_range: if a 2-tuple, the robot footprint will be rotated over this range;
            the returned footprint results from superimposing the footprint at each angle.
            If a single number, a single footprint at that angle will be returned
        robot_footprint: n x 2 numpy array with ROS-style footprint (x, y coordinates),
            in metric units, oriented at 0 angle
        map_resolution:
        :param angle Float: orientation of the robot
        :param robot_footprint array(N, 2)[float64]: n x 2 numpy array with ROS-style footprint (x, y coordinates),
            in metric units, oriented at 0 angle
        :param map_resolution Float: length in metric units of the side of a pixel
        :param fill bool: if True, the footprint will be solid; if False, only the contour will be traced
        :return array(K, M)[uint8]: image of the footprint drawn on the image in white
        '''
        assert not isinstance(angle, tuple)
        angles = [angle]
        m = np.empty((2, 2, len(angles)))  # stack of 2 x 2 matrices to rotate the footprint across all desired angles
        c, s = np.cos(angles), np.sin(angles)
        m[0, 0, :], m[0, 1, :], m[1, 0, :], m[1, 1, :] = (c, -s, s, c)
        rot_pix_footprints = np.rollaxis(np.dot(robot_footprint / map_resolution, m), -1)  # n_angles x n_footprint_corners x 2
        # From all the possible footprints, get the outer corner
        footprint_corner = np.maximum(np.amax(rot_pix_footprints.reshape(-1, 2), axis=0),
                                      -np.amin(rot_pix_footprints.reshape(-1, 2), axis=0))
        pic_half_size = np.ceil(footprint_corner).astype(np.int32)
        int_footprints = np.round(rot_pix_footprints).astype(np.int32)
        # get unique int footprints to save time; using http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
        flat_int_footprints = int_footprints.reshape(len(angles), -1)
        row_view = np.ascontiguousarray(flat_int_footprints).view(np.dtype((np.void, flat_int_footprints.dtype.itemsize * flat_int_footprints.shape[1])))
        _, idx = np.unique(row_view, return_index=True)
        unique_int_footprints = int_footprints[idx]
        kernel = np.zeros(2 * pic_half_size[::-1] + 1, dtype=np.uint8)
        for f in unique_int_footprints:
            if fill:
                cv2.fillPoly(kernel, [f + pic_half_size], (255, 255, 255))
            else:
                cv2.polylines(kernel, [f + pic_half_size], 1, (255, 255, 255))
        return kernel


def ensure_float_numpy(data):
    """
    Transforms python list or tuple to numpy or makes sure that numpy array is float32 or 64
    :param data: a python list, tuple or numpy array.
    :return np.ndarray: An array that is floats for sure.
    """
    if isinstance(data, (list, tuple)):
        return np.array(data, dtype=float)
    else:
        assert(data.dtype in [np.float, np.float32])
        return data


def refine_path(data, delta, angle_delta=None):
    '''
    Insert points in the path if distance between existing points along a path is larger than delta.
    There are two ways to insert points given a constraint on a distance delta:
    Imagine path is from 0 to 1 and constraint is 0.49.
     - option 1:
        step with delta from the begging till the end:
        (0, 0.49, 0.98, 1)
        most of the time distance is equal to delta, but there might be really short steps like the last one
     - option 2:
        determine how many points to insert and make equal steps (1/0.49 + 1 = 3 intervals
        (0, 0.333, 0.666, 1)
        distance between points is smaller than delta, but there are no really small steps

    This function uses option 2.

    Angle is interpolated if angle_delta is not None. Interpolation happens
        by copying the second point in a consecutive pair several times
        with different angles.

    :param data: a np array of n x (x, y) or (x, y, angle) elements
    :param delta: maximum distance between points
    :param angle_delta: if not None, angle differences bigger than this will be
        interpolated; interpolation happens by copying the second point in
        a consecutive pair several times with different angles. Angle must be
        provided if not None.
    :returns regularized_path: a np array of m x (x, y) or (x, y, angle) elements
    '''
    if isinstance(data, (list, tuple)):
        data = np.array(data, dtype=float)
    else:
        assert(data.dtype in [np.float, np.float32])
    if data.shape[1] not in (2, 3):
        raise Exception("This function takes n x (x, y) or n x (x, y, angle) arrays")
    if angle_delta is not None and data.shape[1] != 3:
        raise Exception("Path does not include angles but angle interpolation was requested")
    xy = data[:, :2]
    segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if angle_delta is not None:
        angles = diff_angles(data[1:, 2], data[:-1, 2])
    pieces = []
    for i, d in enumerate(segment_lengths):
        if d > delta:
            # interpolate all coordinates one by one (e.g. x, y)
            npoints = int(d / delta) + 2
            interpolated = [np.linspace(data[i, j], data[i+1, j], num=npoints)
                            for j in range(2)]
            # copy angle between interpolations
            if data.shape[1] == 3:
                interpolated.append(np.ones((npoints,), dtype=float)*data[i, 2])
            piece = np.vstack(interpolated).T
            # cut down the end in order to join with the next piece
            pieces.append(piece[:-1])
        else:
            pieces.append(data[i])
        if angle_delta is not None and np.abs(angles[i]) > angle_delta:
            n_points = int(np.abs(angles[i]) / angle_delta) + 1
            interpolated_angles = normalize_angle(data[i, 2] + angles[i] / n_points * np.arange(n_points))
            pieces.append([[data[i + 1, 0], data[i + 1, 1], a] for a in interpolated_angles])

    # add path endpoint
    pieces.append(data[-1])
    return np.vstack(pieces)


def orient_path(path, robot_pose=None, final_pose=None, max_movement_for_turn_in_place=0.0):
    """ This function orient the path
    For those points have large movement (not turn-in-place points), use the diff angle as the pose heading. For those
    turn-in-place points, use not turn-in-places points before and after to interpolate.

    Final pose is only used to decide the pose angle so that it's guaranted that the position of oriented_path and
    path are exactly the same.

    :param path: np.ndarray: N * 2 path points
    :param robot_pose: np.ndarray: 3 * 1 (x, y, theta), current robot pose
    :param final_pose: np.ndarray: 3 * 1 (x, y, theta), desired last pose
    :param max_movement_for_turn_in_place: float: points within this distance are considered as turn-in-place
    :return: np.ndarray: N * 3 path with orientation
    """
    path = ensure_float_numpy(path)

    # keep the position the same
    oriented_path = np.zeros((len(path), 3), dtype=path.dtype)
    if len(path) == 0:
        return oriented_path
    assert path.shape[1] == 2, 'The shape[1] of path is {}, which is not 2'.format(path.shape[1])
    oriented_path[:, :2] = path[:, :2]
    if len(path) == 1:
        oriented_path[0, 2] = 0.
        return oriented_path

    path_diff = path[1:, :] - path[0: -1, :]
    path_angles = np.arctan2(path_diff[:, 1], path_diff[:, 0])

    # if given final pose, use it as last pose
    oriented_path[:-1, 2] = path_angles
    oriented_path[-1, 2] = final_pose[2] if final_pose is not None else oriented_path[-2, 2]

    turn_in_place_points = np.hypot(path_diff[:, 0], path_diff[:, 1]) < max_movement_for_turn_in_place
    if robot_pose is not None and turn_in_place_points[0]:
        oriented_path[0, 2] = robot_pose[2]
        turn_in_place_points[0] = False

    # TODO: optimize this
    left_not_turn_in_place_indices = np.zeros(len(path_angles), dtype=int)
    for i in range(1, len(left_not_turn_in_place_indices)):
        left_not_turn_in_place_indices[i] = left_not_turn_in_place_indices[i-1] if turn_in_place_points[i] else i
    right_not_turn_in_place_indices = np.zeros(len(path_angles), dtype=int)
    right_not_turn_in_place_indices[-1] = len(path_angles)
    for i in range(len(path_angles)-2, -1, -1):
        right_not_turn_in_place_indices[i] = right_not_turn_in_place_indices[i+1] if turn_in_place_points[i] else i

    for i in np.where(turn_in_place_points)[0]:
        left_index = left_not_turn_in_place_indices[i]
        right_index = right_not_turn_in_place_indices[i]
        oriented_path[i, 2] = (oriented_path[left_index, 2] * (right_index - i) +
                               oriented_path[right_index, 2] * (i - left_index)) / (right_index - left_index)
    return oriented_path


def path_velocity(path):
    """
    Compute velocities of the path.
    :param path: n x 4 array of (t, x, y, angle)
    :return: arrays of v and w along the path
    """
    path = np.asarray(path)
    diffs = np.diff(path, axis=0)
    dt = diffs[:, 0]
    assert (dt > 0).all()

    # determine direction of motion
    dxy = diffs[:, 1:3]
    sign = np.sign(np.cos(path[:, 3])[:-1]*dxy[:, 0] + np.sin(path[:, 3])[:-1]*dxy[:, 1])
    sign[sign == 0.] = np.sign(np.sin(path[:, 3])[:-1]*dxy[:, 1])[sign == 0.]

    ds = np.linalg.norm(dxy, axis=1)*sign
    dangle = diffs[:, 3]
    dangle[dangle < -np.pi] += 2*np.pi
    dangle[dangle > np.pi] -= 2*np.pi
    # large angular velocities are not supported
    if not (np.abs(dangle) < np.pi).all():
        bad_indices = (np.abs(dangle) > np.pi).nonzero()
        raise Exception("Path has missing/corrupted angle data at indices: %s. Data: %s" %
                        (bad_indices, dangle[bad_indices]))
    return ds/dt, dangle/dt


def draw_arrow(image, p, q, color, arrow_magnitude=5, thickness=1, line_type=8, shift=0):
    """Draw an arrow on an image.
    :param image np.ndarray(h, w, c): image to draw the image onto
    :param p Tuple[int, int]: arrows goes from this point
    :param q Tuple[int, int]: arrows goes to this point
    :param color Tuple[int, int, int]: of this color
    :param arrow_magnitude int: how prominent the arrow shuold be
    :param thickness int: thickness of the arrow
    :param line_type int: what is cv2 line type
    :param shift int: should arrow be shifted
    """
    # adapted from http://mlikihazar.blogspot.com.au/2013/02/draw-arrow-opencv.html
    # draw arrow tail
    p = (int(p[0]), int(p[1]))
    q = (int(q[0]), int(q[1]))
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # calc angle of the arrow
    angle = np.arctan2(p[1]-q[1], p[0]-q[0])
    # starting point of first line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),
         int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
    # draw first half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # starting point of second line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),
         int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
    # draw second half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)


def pose_distances(pose0, pose1):
    """
    Determine linear and angular difference between poses
    :param pose0: (x, y, angle) or array of pose number one
    :param pose1: (x, y, angle) or array of pose number two
    :return: linear and angular distances
    """
    assert isinstance(pose0, np.ndarray)
    assert isinstance(pose1, np.ndarray)
    assert pose0.shape == pose1.shape

    return np.hypot(pose0[..., 0] - pose1[..., 0], pose0[..., 1] - pose1[..., 1]),\
        np.abs(diff_angles(pose0[..., 2], pose1[..., 2]))


def limit_path_index(path, max_dist, max_angle=np.inf, min_length=0):
    """
    From the given path take only the number of points that will not exceed
    the distance or angle criterion (whichever is less).
    :param path np.ndarray(n, 3): (x, y, theta) of the path
    :param max_dist: max distance along the path
    :param max_angle: max rotation along the path
    :param min_length: don't cut shorter than that

    :return: cutoff index that should *NOT* be included in the path.
    """
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return len(path)

    d_path = np.diff(path, axis=0)

    cum_dist = np.cumsum(np.hypot(d_path[:, 0], d_path[:, 1]))
    cum_angle = np.cumsum(np.abs(np.angle(np.exp(1j * d_path[:, 2]))))
    min_dist_cutoff = len(path) if cum_dist[-1] <= min_length else (1 + np.argmax(cum_dist > min_length))
    dist_cutoff = len(path) if cum_dist[-1] <= max_dist else (1 + np.argmax(cum_dist > max_dist))
    angle_cutoff = len(path) if cum_angle[-1] <= max_angle else (1 + np.argmax(cum_angle > max_angle))
    cutoff = min(len(path), max(min_dist_cutoff, min(dist_cutoff, angle_cutoff)))
    return cutoff


def parallel_distances(pose, path_poses):
    """
    Signed projection of the vector from path_poses to pose along the
    direction given by path_pose orientation.
    :param pose np.ndarray(3): (x, y, theta) of the robot
    :param path_poses np.ndarray(n, 3): (x, y, theta) of the path
    :return np.ndarray(n): the projection distances
    """
    return np.cos(path_poses[:, 2]) * (pose[0] - path_poses[:, 0]) + np.sin(path_poses[:, 2]) * (pose[1] - path_poses[:, 1])


def find_reached_indices(pose, segment, spatial_precision, angular_precision, parallel_distance_threshold=None):
    """
    Walk along the path and determine which point we have reached; a point
    is reached if it is so either wrt the original path or the
    deformed path.

    :param pose: (x, y, theta) of the robot
    :param segment: m x 3 points corresponding to the segment deformed by the elastic planner
    :param spatial_precision float: spatial precision
    :param angular_precision float: angular precision
    :param parallel_distance_threshold float float: maximum allowed parallel distance
    :return: all reached indices
    """
    assert len(segment) > 0
    if parallel_distance_threshold is None:
        parallel_distance_threshold = -spatial_precision / 9
    dist = np.hypot(segment[:, 0] - pose[0], segment[:, 1] - pose[1])
    angle = np.abs(diff_angles(pose[2], segment[:, 2]))
    parallel_dist = parallel_distances(pose, segment)
    reached_idx = np.where(np.logical_and(np.logical_and(dist < spatial_precision, angle < angular_precision),
                                          parallel_dist >= parallel_distance_threshold))[0]
    return reached_idx


def find_last_reached(pose, segment, spatial_precision, angular_precision, parallel_distance_threshold=None):
    """
    Walk along the path and determine which point we have reached; a point
        is reached if it is so either wrt the original path or the
        deformed path.

    :param pose: (x, y, theta) of the robot
    :param segment np.ndarray(m, 3): m x 3 points corresponding to the segment deformed by the elastic planner
    :param spatial_precision float: spatial precision
    :param angular_precision float: angular precision
    :param parallel_distance_threshold float float: maximum allowed parallel distance
    :return: the max index in the segment that is reached
    """
    reached_idx = find_reached_indices(pose, segment, spatial_precision, angular_precision, parallel_distance_threshold)
    if len(reached_idx):
        return reached_idx[-1]
    return None


def distance_to_segments(point, segment_origins, segment_ends):
    """
    Returns the shortest distance from the given point to a sequence
    of segments defined by (n x 2) matrices segment_origins and segment_ends.
    :param point np.ndarray(2)): the point to measure
    :param segment_origins np.ndarray(n, 2): beginnings of the segments in 2d
    :param segment_ends np.ndarray(n, 2): endings of the segments in 2d
    :return float: the closest distance to some segment from give segment list
    """
    segment_ends = np.asarray(segment_ends)
    segment_origins = np.asarray(segment_origins)
    v = segment_ends - segment_origins
    v_norm_square = np.sum(v * v, axis=1)
    # mu will be a scalar determining the position of the projection of point on each segment
    # mu is normalized so that 0 is at the segment origin, 0.5 the midpoint, 1 the segment end
    mu = np.sum((point - segment_origins) * v, axis=1) / (v_norm_square + 1e-12)
    # clip mu to [0, 1]: if the projection is outside the segment, the closest point is the vertex
    mu = np.clip(mu, 0., 1.)
    closest_points = segment_origins + mu[:, None] * v
    return np.hypot(point[0] - closest_points[:, 0], point[1] - closest_points[:, 1])


def distances_to_segment(points, segment_origin, segment_end):
    '''
    Returns the shortest distance from the given points, defined by an
    n x 2 matrix, to the segment defined by 2-vectors segment_origin and segment_end.
    :param points array(N, 2)[float]: (x, y) array of points
    :param segment_origin array(2)[float]: (x, y) origin of a segment
    :param segment_end array(2)[float]: (x, y) end of a segment
    :return array(N)[float]: distances
    '''
    segment_end = np.asarray(segment_end)
    segment_origin = np.asarray(segment_origin)
    v = segment_end - segment_origin
    v_norm_square = np.sum(v * v)
    # mu will be a scalar determining the position of the projection of point on each segment
    # mu is normalized so that 0 is at the segment origin, 0.5 the midpoint, 1 the segment end
    mu = np.sum((points - segment_origin) * v, axis=1) / (v_norm_square + 1e-12)
    # clip mu to [0, 1]: if the projection is outside the segment, the closest point is the vertex
    mu = np.clip(mu, 0., 1.)
    closest_points = segment_origin + mu[:, None] * v
    return np.hypot(points[:, 0] - closest_points[:, 0], points[:, 1] - closest_points[:, 1])


def distances_to_multiple_segments(point, segment_origins, segment_ends):
    '''
    Returns the shortest distances from the given point (x, y), to the segments defined by nx2-vectors segment_origins and segment_ends.
    :param point array(2)[float]: (x, y) a point
    :param segment_origins array(N, 2)[float]: array of (x, y) origins of segments
    :param segment_ends array(N, 2)[float]: array of (x, y) ends of segments
    :return array(N)[float]: distances
    '''
    point = np.asarray(point)
    segment_origins = np.asarray(segment_origins)
    segment_ends = np.asarray(segment_ends)
    assert segment_origins.shape == segment_ends.shape
    assert point.shape == (2,)
    v = segment_ends - segment_origins
    v_norm_squares = np.sum(v * v, axis=1)
    # mu will be a scalar determining the position of the projection of point on each segment
    # mu is normalized so that 0 is at the segment origin, 0.5 the midpoint, 1 the segment end
    mu = np.sum((point - segment_origins) * v, axis=1) / (v_norm_squares + 1e-12)
    # clip mu to [0, 1]: if the projection is outside the segment, the closest point is the vertex
    mu = np.clip(mu, 0., 1.)
    closest_points = segment_origins + mu[:, None] * v
    return np.hypot(point[0] - closest_points[:, 0], point[1] - closest_points[:, 1])


def inscribed_radius(footprint):
    '''
    Returns the shortest distance from (0, 0) to any of the sides
    of a footprint (n x 2 numpy array of consecutive vertices)
    :param footprint array(n, 2)[float]: n x 2 numpy array with ROS-style footprint (x, y coordinates),
        in metric units, oriented at 0 angle
    :return float: inscribed radius
    '''
    segment_ends = np.roll(footprint, -1, axis=0)
    return fast_amin(distance_to_segments((0, 0), footprint, segment_ends))


def circumscribed_radius(footprint):
    '''
    Returns the longest distance from (0, 0) to any of the sides
    of a footprint (n x 2 numpy array of consecutive vertices)
    :param footprint array(n, 2)[float]: n x 2 numpy array with ROS-style footprint (x, y coordinates),
        in metric units, oriented at 0 angle
    :return float: circumscribed radius
    '''
    return fast_amax(np.linalg.norm(footprint, axis=1))


def get_pixel_in_map_mask(map_shape, pixels):
    """
    Filter for pixels that fall inside the costmap.
    :param map_shape: (size_y, size_x) of costmap
    :param pixels: array(N, 2) of (x, y) pixel coordinates
    :return: array(N)[bool] which can be used to mask in pixels that are inside the map.
    """
    ixs = (pixels[:, 1] >= 0) & (pixels[:, 1] < map_shape[0]) & (pixels[:, 0] >= 0) & (pixels[:, 0] < map_shape[1])
    return ixs


def compute_robot_area(resolution, robot_footprint):
    """
    Computes robot footprint area in pixels
    :param resolution float: resolution of the costmap
    :param robot_footprint array(N, 2)[float]: footprint polygon
    :return float: area of robot's footprint in pixels
    """
    return float(np.count_nonzero(get_pixel_footprint(0., robot_footprint, resolution)))
