from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


from builtins import range
from builtins import zip
import numpy as np
import pytest


from bc_gym_planning_env.utilities.coordinate_transformations import normalize_angle, rotation_matrix, \
    homogenize, de_homogenize, project_pose, from_homogeneous_matrix, \
    transform_to_homogeneous_matrix, project_points, from_rotation_matrix, add_transforms, \
    chain_transforms, add_transforms_batch, transform_pose, transforms_between_poses, transform_between_poses, \
    transforms_between_pose_pairs, compose_covariances, compose_covariances_batch, add_uncertain_transforms, \
    check_valid_cov_mat, normalize_transform_angle, cumsum_uncertain_transforms, rigid_body_pose_transform, \
    rigid_body_pose_transforms_batch, transformation_between_coordinate_frames, \
    inverse_transform, project_poses, \
    from_global_to_egocentric, from_egocentric_to_global, transform_poses_batch, get_map_to_odom_tranformation, \
    from_map_to_odom, interpolate_angles, world_to_pixel


def normalize_angle_reference(z):
    '''
    Normalize angles to -pi to pi
    # http://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap
    '''
    return (np.array(z) + np.pi) % (2 * np.pi) - np.pi


def test_normalize_angles():
    np.testing.assert_array_almost_equal(
        normalize_angle(0.),
        0.
    )

    np.testing.assert_array_almost_equal(
        normalize_angle(np.pi/2),
        np.pi/2
    )

    np.testing.assert_array_almost_equal(
        normalize_angle(-np.pi/2),
        -np.pi/2
    )

    np.testing.assert_array_almost_equal(
        normalize_angle(np.pi+0.1),
        -np.pi+0.1
    )

    np.testing.assert_array_almost_equal(
        normalize_angle(-np.pi-0.1),
        np.pi-0.1
    )

    np.testing.assert_array_almost_equal(
        normalize_angle(np.array([0., 1, -2])),
        [0., 1, -2]
    )

    np.testing.assert_array_almost_equal(
        normalize_angle(np.array([0., np.pi + 0.1])),
        [0., -np.pi + 0.1]
    )

    np.testing.assert_array_almost_equal(
        normalize_angle(np.array([-np.pi - 0.1, 2 * np.pi + 0.1, 99 * np.pi + 0.1, 100 * np.pi + 0.1, -1001 * np.pi - 0.3])),
        [np.pi - 0.1, 0.1, -np.pi + 0.1, 0.1, np.pi - 0.3]
    )

    np.testing.assert_array_almost_equal(normalize_angle(np.pi), normalize_angle_reference(np.pi))
    np.testing.assert_array_almost_equal(normalize_angle(-np.pi), normalize_angle_reference(-np.pi))

    rand_angle_scalars = np.random.randn(100)*10. - 5
    for a in rand_angle_scalars:
        a = float(a)
        np.testing.assert_array_almost_equal(normalize_angle(a), normalize_angle_reference(a))

    rand_angles = np.random.randn(1000, 100)*10. - 5
    for a in rand_angles:
        np.testing.assert_array_almost_equal(normalize_angle(a), normalize_angle_reference(a))


try:
    # import cpp optimized implementation if possible and run the test only then
    from brain.shining_utils.transform_utils import normalize_angle_impl  # pylint: disable=unused-import

    def test_angle_normalize_pybind_checks():
        """Test how does angle normalization handles non-contiguous input (should raise) or 2d input"""
        data = np.array([[0., np.pi + 0.1],
                         [0., 0.2],
                         [0., 2*np.pi+0.3]])
        with pytest.raises(TypeError):
            normalize_angle(data)
        non_contiguous_data = data[:, 1]
        assert not non_contiguous_data.flags['C_CONTIGUOUS']
        with pytest.raises(TypeError):
            normalize_angle(non_contiguous_data)
except ImportError:
    pass


def test_inverse_transform():
    # Translate around loop (translate, then rotate)
    t_b_a = np.array([1., 23., np.pi / 2.])

    # res is t_a_a, so should be (0, 0, 0)
    np.testing.assert_array_almost_equal(
        add_transforms(t_b_a, inverse_transform(t_b_a)),
        [0., 0., 0.]
    )

    np.testing.assert_array_almost_equal(
        inverse_transform(np.array([0., 0., 0.])),
        [0., 0., 0.]
    )

    np.testing.assert_array_almost_equal(
        inverse_transform(np.array([-0.1, 0.2, 0.])),
        [0.1, -0.2, 0.]
    )

    np.testing.assert_array_almost_equal(
        inverse_transform(np.array([1., 0., -np.pi / 2.])),
        [0., -1., np.pi / 2.]
    )

    np.testing.assert_array_almost_equal(
        inverse_transform(np.array([1., 1., np.pi/4])),
        [-np.sqrt(2), 0., -np.pi/4]
    )

    np.testing.assert_array_almost_equal(
        inverse_transform(np.array([0., 0., 2*np.pi])),
        [0., 0., 0.]
    )


def test_inverse_transform_batch():
    transforms = [[1., 23., np.pi / 2.],
                  [0., 0., 0.],
                  [-0.1, 0.2, 0.],
                  [1., 1., np.pi / 4]]
    transforms = np.array(transforms)

    np.testing.assert_array_almost_equal(
        inverse_transform(transforms),
        [[-23, 1., -np.pi / 2.],
         [0., 0., 0.],
         [0.1, -0.2, 0.],
         [-np.sqrt(2), 0., -np.pi / 4]]
    )


def inverse_transform_reference(transform):
    """
    Numpy reference inverse_transform
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


def test_inverse_transform_consistency():
    transforms = np.random.rand(1000, 3)*10. - 5.
    for t in transforms:
        np.testing.assert_array_almost_equal(inverse_transform(t), inverse_transform_reference(t))


def test_rotation_matrix():
    np.testing.assert_array_almost_equal(
        rotation_matrix(0.),
        [[1., 0.],
         [0., 1.]]
    )

    np.testing.assert_array_almost_equal(
        rotation_matrix(-np.pi/4),
        [[np.sqrt(2.)/2., np.sqrt(2.)/2.],
         [-np.sqrt(2.)/2., np.sqrt(2.)/2.]]
    )


def test_from_homogeneous_matrix():

    np.testing.assert_array_almost_equal(
        from_homogeneous_matrix(np.array([[1., 0., 3.],
                                          [0., 1., -13.],
                                          [0., 0., 1]])),
        [3., -13., 0.]
    )

    np.testing.assert_array_almost_equal(
        from_homogeneous_matrix(np.array([[np.cos(-np.pi/3), -np.sin(-np.pi/3), 4.],
                                          [np.sin(-np.pi/3), np.cos(-np.pi/3), -12.],
                                          [0., 0., 1]])),
        [4., -12., -np.pi/3]
    )

    for _ in range(50):
        t = np.random.rand(3) * np.pi - np.pi / 2.
        np.testing.assert_array_almost_equal(t, from_homogeneous_matrix(transform_to_homogeneous_matrix(t)))


def test_transform_as_matrix_composition():
    for _ in range(50):
        t0 = np.random.rand(3) * np.pi - np.pi / 2.
        t1 = np.random.rand(3) * np.pi - np.pi / 2.

        result = add_transforms(t0, t1)

        matrix = np.dot(transform_to_homogeneous_matrix(t0), transform_to_homogeneous_matrix(t1))
        np.testing.assert_array_almost_equal(from_homogeneous_matrix(matrix), result)


def test_trivial_transform_composition():
    """
    Test composition operator for transforms
    """

    t_a_a = np.array([0., 0., 0.])
    t_b_a = np.array([1., 0., 0.])

    # Translate 1.0 meters forward
    res = add_transforms(t_a_a, t_b_a)

    assert res[0] == 1.0
    assert res[1]== 0.0
    assert res[2] == 0.0


def test_transform_composition_1():
    """
    Half diamond turn
    """
    t_b_a = np.array([1., 1., np.pi / 2.])
    t_c_b = np.array([1., 1., np.pi / 2.])
    np.testing.assert_array_almost_equal(
        add_transforms(t_b_a, t_c_b),
        [0., 2., -np.pi]
    )


def test_transform_composition_2():
    """
    Straight forward, rotate 90 degrees clockwise, then
    forward 1.0 meters and rotate  90 degrees counterclockwise
    """

    t_b_a = np.array([1., 1., np.pi / 2.])
    t_c_b = np.array([1., 0., -np.pi / 2.])

    res = add_transforms(t_b_a, t_c_b)

    tol = 1e-6

    assert np.abs(res[0] - 1.0) < tol
    assert np.abs(res[1] - 2.0) < tol
    assert np.abs(res[2] - 0.) < tol


def test_perfect_transform_loop():
    """
    Transform around a perfect square loop
    """

    # Translate around loop (translate, then rotate)
    t_b_a = np.array([1., 0., np.pi / 2.])
    t_c_b = np.array([1., 0., np.pi / 2.])
    t_d_c = np.array([1., 0., np.pi / 2.])
    t_a_d = np.array([1., 0., np.pi / 2.])

    # res is t_a_a, so should be (0, 0, 0)
    res = add_transforms(add_transforms(add_transforms(t_b_a, t_c_b), t_d_c), t_a_d)

    tol = 1e-6
    assert np.abs(res[0]) < tol
    assert np.abs(res[1]) < tol
    assert np.abs(np.mod(res[2], np.pi*2)) < tol


def test_homogenize():
    point = np.zeros((1, 2))
    res = homogenize(point)
    assert res[0, 2] == 1.0
    assert res.shape == (1, 3)

    points = np.array([[1., 2.],
                       [-1., 78.]
                       ])
    res = homogenize(points)

    assert res[0, 2] == 1.0
    assert res.shape == (2, 3)


def test_de_homogenize():
    points = np.array([[1., 2., 1.0],
                       [-1., 78., 1.0]
                       ])
    res = de_homogenize(points)

    target = np.array([[1., 2.],
                       [-1., 78]
                       ])
    assert res.shape == (2, 2)
    np.testing.assert_array_equal(target, res)


def test_homogenize_de_homogenize():
    points = np.array([[1., 2.],
                       [-1., 78.]
                       ])
    res = homogenize(points)
    res = de_homogenize(res)

    assert res.shape == (2, 2)
    np.testing.assert_array_equal(points, res)


def test_compose_matrix():
    """
    Test transform_to_homogeneous_matrix functionality
    """

    # Test translation only
    point = np.zeros((1, 2))
    m = transform_to_homogeneous_matrix([1.0, 0.0, 0.])
    result = de_homogenize(np.dot(m, homogenize(point).T).T)

    target = np.array([[1.0, 0.0]])
    np.testing.assert_array_almost_equal(result, target)

    # Test rotation only
    point = np.array([[1.0, 0.0]])
    m = transform_to_homogeneous_matrix([0., 0., np.pi / 2.])
    result = de_homogenize(np.dot(m, homogenize(point).T).T)

    target = np.array([[0.0, 1.0]])
    np.testing.assert_array_almost_equal(result, target)

    # Test rotation + translation
    point = np.array([[1.0, 0.0]])
    m = transform_to_homogeneous_matrix([1.0, 0.0, np.pi / 2.])
    result = de_homogenize(np.dot(m, homogenize(point).T).T)

    target = np.array([[1.0, 1.0]])
    np.testing.assert_array_almost_equal(result, target)

    # Make sure that M M^I ends up as I
    m = transform_to_homogeneous_matrix([1.0, 32.0, np.pi / 7.])
    np.testing.assert_array_almost_equal(np.dot(m, np.linalg.inv(m)), np.identity(3))


def test_homogeneous_matrix():
    """
    Use transform to project between frames (using homogeneous matrix form)
    """
    # Child frame transform in parent frame
    C = np.array([10., 2., np.pi / 2.])

    # Interpreting C as a homogeneous transform matrix
    # tells us how to transform points in C to W (the parent frame)
    m = transform_to_homogeneous_matrix(C)

    # A point at the origin of C should stay there
    point = np.zeros((1, 2))  # in C coordinate frame
    result = de_homogenize(np.dot(m, homogenize(point).T).T)

    np.testing.assert_array_almost_equal(result, np.array([[10., 2.]]))

    # A point in front of C is "up" in W
    point = np.array([[1., 0]])  # in C coordinate frame
    result = de_homogenize(np.dot(m, homogenize(point).T).T)

    np.testing.assert_array_almost_equal(result, np.array([[10., 3.]]))

    # Projecting from Parent to C(hild) requires the inverse:
    m = transform_to_homogeneous_matrix(inverse_transform(C))

    # Projecting a point at Parent origin to Child coordinate frame (note the rotation!)
    point = np.zeros((1, 2))  # in C coordinate frame
    result = de_homogenize(np.dot(m, homogenize(point).T).T)
    np.testing.assert_array_almost_equal(result, np.array([[-2., 10.]]))


def test_project_points():
    # Child frame transform in parent frame
    C = np.array([10., 2., np.pi / 2.])

    # A point at the origin of C should stay there
    points = np.zeros((1, 2))
    np.testing.assert_array_almost_equal(
        project_points(C, points),
        np.array([[10., 2.]])
    )

    # A point in front of C is "up" in W
    points = np.array([[1., 0]])
    np.testing.assert_array_almost_equal(
        project_points(C, points),
        np.array([[10., 3.]]))

    # Projecting from Parent to C(hild) requires the inverse:
    # Projecting a point at Parent origin to Child coordinate frame (note the rotation!)
    points = np.zeros((1, 2))  # in C coordinate frame
    np.testing.assert_array_almost_equal(
        project_points(inverse_transform(C), points),
        np.array([[-2., 10.]])
    )


def test_project_no_points():
    assert len(project_points(np.array([0.1, 0.2, 0.3]), np.empty((0, 2)))) == 0


def test_project_single_point():

    def _test_single_point_transform(point, transform, expected_point):
        np.testing.assert_array_almost_equal(
            project_points(
                np.array(transform),
                np.array([point])
            ),
            np.array([expected_point])
        )

    # dummy transforms
    _test_single_point_transform(
        point=[1., -2.],
        transform=[0., 0., 0],
        expected_point=[1., -2.])

    _test_single_point_transform(
        point=[1., 0.],
        transform=[1., 0., 0],
        expected_point=[2., 0.])

    # child frame is higher than parent. Point in parent frame becomes higher
    _test_single_point_transform(
        point=[1., 0.],
        transform=[0., 1., 0],
        expected_point=[1., 1.])

    # point in a child frame pointed up
    _test_single_point_transform(
        point=[1., 0.],
        transform=[0., 0., np.pi / 2.],
        expected_point=[0., 1.])

    # point in a child frame pointed up and shifted back
    _test_single_point_transform(
        point=[1., 0.],
        transform=[-1., 0., np.pi / 2.],
        expected_point=[-1., 1.])


def test_transforms_between_poses():
    np.testing.assert_array_almost_equal(
        transforms_between_poses(np.array([
            [0., 0., 0.],
            [0., 0., 0.]
        ])),
        [[0., 0., 0.]]
    )

    np.testing.assert_array_almost_equal(
        transforms_between_poses(np.array([
            [0., 0., 0.],
            [1., 0., 0.]
        ])),
        [[1., 0., 0.]]
    )

    np.testing.assert_array_almost_equal(
        transforms_between_poses(np.array([
            [0., -2., 0.],
            [1., 0., 0.]
        ])),
        [[1., 2., 0.]]
    )

    np.testing.assert_array_almost_equal(
        transforms_between_poses(np.array([
            [0., 0., 0.],
            [0., 0., np.pi/4]
        ])),
        [[0., 0., np.pi/4]]
    )

    np.testing.assert_array_almost_equal(
        transforms_between_poses(np.array([
            [0., 0., np.pi / 2.],
            [1., 0., 0.]
        ])),
        [[0., -1., -np.pi / 2.]]
    )

    def _transforms_from_path(path):
        # older version of the function
        transforms = []
        for i in range(len(path) - 1):
            t = add_transforms(
                inverse_transform(path[i]), path[i + 1])
            transforms.append(t)
        return np.array(transforms)

    for _ in range(100):
        pose0 = normalize_transform_angle((np.random.random_sample(3) - 0.5) * 20)
        pose1 = normalize_transform_angle((np.random.random_sample(3) - 0.5) * 20)
        pose2 = normalize_transform_angle((np.random.random_sample(3) - 0.5) * 20)

        path = np.array([pose0, pose1, pose2])
        transforms = transforms_between_poses(path)
        assert transforms.shape == (2, 3)

        transforms_alt = _transforms_from_path(path)
        np.testing.assert_array_almost_equal(transforms, transforms_alt)

        np.testing.assert_array_almost_equal(
            transform_pose(pose0, transforms[0]),
            pose1
        )

        np.testing.assert_array_almost_equal(
            transform_pose(pose1, transforms[1]),
            pose2
        )

        np.testing.assert_array_almost_equal(
            transform_pose(pose0, transforms[0], transforms[1]),
            pose2
        )


def test_correct_dehomogenize():

    # Carry out a basic perspective transform:
    mat = np.array([[1., 0, 0], [0, 1, 0], [0, 1, 1]])
    x = np.array([[1, 0], [1., 1.], [1, 2]])
    yh = mat.dot(homogenize(x).T).T
    y = de_homogenize(yh)
    assert np.allclose(y, [[1.0, 0.0], [0.5, 0.5], [1./3, 2./3]])
    assert not np.allclose(y, yh[:, :2])  # This would have failed with the previous (buggy) de_homogenize function.


def test_project_points_and_transform_relations():
    poseA = np.array([0., 1., np.pi / 2.])
    poseB = np.array([1., 0., 0.])

    transform = transforms_between_poses(np.array([
        poseA,
        poseB,
    ]))[0]

    point_in_B = np.array([2., -1])
    point_in_global = project_points(poseB, point_in_B[None, :])[0]
    np.testing.assert_array_almost_equal(point_in_global, [3., -1])

    point_in_A = project_points(transform, point_in_B[None, :])[0]
    np.testing.assert_array_almost_equal(point_in_A, [-2, -3])

    np.testing.assert_array_almost_equal(
        project_points(poseA, point_in_A[None, :])[0],
        point_in_global
    )


def test_project_points_and_transform_relations_2():
    '''
    Check that projecting points twice is the same as projecting once via added transforms
    '''
    poseA = np.array([0., 1., np.pi / 2.])
    poseB = np.array([1., 0., 0.])
    poseC = np.array([3., 0., -np.pi / 2.])

    transformAB, transformBC = transforms_between_poses(np.array([
        poseA,
        poseB,
        poseC
    ]))

    points_in_C = np.array([[2., -1], [2., 1]])
    points_in_B = project_points(transformBC, points_in_C)

    np.testing.assert_array_almost_equal(points_in_B, [[1., -2], [3., -2]])

    points_in_A = project_points(transformAB, points_in_B)
    np.testing.assert_array_almost_equal(points_in_A, [[-3., -2], [-3., -4]])

    np.testing.assert_array_almost_equal(
        points_in_A,
        project_points(add_transforms(transformAB, transformBC), points_in_C)
    )

    for _ in range(20):
        poseA = (np.random.random_sample(3) - 0.5) * 10
        poseB = (np.random.random_sample(3) - 0.5) * 10
        poseC = (np.random.random_sample(3) - 0.5) * 10

        transformAB, transformBC = transforms_between_poses(np.array([
            poseA,
            poseB,
            poseC
        ]))

        # here is a explicit relation
        np.testing.assert_array_almost_equal(
            project_points(transformAB, project_points(transformBC, points_in_C)),
            project_points(add_transforms(transformAB, transformBC), points_in_C)
        )


def test_compose_identity_covariances():
    c0 = np.identity(3)
    t0 = np.array([0., 0., 0.])
    t_rotate = np.array([0., 0., np.pi / 2.])
    t_rotate_revert = np.array([0., 0., np.pi])

    t_shift_right = np.array([1., 0., 0.])
    t_shift_up = np.array([0., 1., 0.])

    # tests with adding covariances
    np.testing.assert_array_almost_equal(
        compose_covariances(t0, c0, t0, c0),
        c0*2  # two covariances just add up
    )

    np.testing.assert_array_almost_equal(
        compose_covariances(t0, c0*0.1, t0, c0*0.2),
        c0 * 0.1 + c0*0.2  # two covariances just add up
    )

    np.testing.assert_array_almost_equal(
        compose_covariances(t_rotate, c0, t0, c0),
        c0 * 2
    )

    np.testing.assert_array_almost_equal(
        compose_covariances(t_rotate_revert, c0, t0, c0),
        c0 * 2
    )

    np.testing.assert_array_almost_equal(
        compose_covariances(t_rotate_revert, c0, t_rotate, c0),
        c0 * 2
    )

    np.testing.assert_array_almost_equal(
        compose_covariances(t_rotate_revert, c0, t_rotate, c0),
        c0 * 2
    )

    np.testing.assert_array_almost_equal(
        compose_covariances(t0, c0, t_shift_right, c0),
        [[2., 0., 0.],
         [0., 3., 1.],
         [0., 1., 2]]
    )

    np.testing.assert_array_almost_equal(
        compose_covariances(t0, c0, t_shift_up, c0),
        [[3., 0., -1.],
         [0., 2., 0.],
         [-1., 0., 2]]
    )

    np.testing.assert_array_almost_equal(
        compose_covariances(t_rotate_revert, c0, t_shift_up, c0),
        [[3., 0., 1.],
         [0., 2., 0.],
         [1., 0., 2]]
    )

    np.testing.assert_array_almost_equal(
        compose_covariances(t0, c0, t_shift_up, c0),
        compose_covariances(t_shift_right, c0, t_shift_up, c0),
    )


def test_compose_covariances():
    c_random_0 = np.array(
        [[0.01432543, 0.00728222, -0.01038915],
         [0.00728222, 0.06027395, 0.00447672],
         [-0.01038915, 0.00447672, 0.01287054]])
    check_valid_cov_mat(c_random_0)
    c_random_1 = np.array(
        [[0.03767351, 0.01022357, 0.00771092],
         [0.01022357, 0.02162842, -0.00759028],
         [0.00771092, -0.00759028, 0.05589302]])
    check_valid_cov_mat(c_random_1)
    t0 = np.array([0., 0., 0.])
    t_rotate = np.array([0., 0., np.pi / 2.])
    t_rotate_revert = np.array([0., 0., np.pi])

    t_shift_complex = np.array([-0.5, 1., -np.pi/5])

    np.testing.assert_array_almost_equal(
        compose_covariances(t0, c_random_0, t0, c_random_1),
        c_random_0 + c_random_1
    )

    np.testing.assert_array_almost_equal(
        compose_covariances(t_rotate, c_random_1, t0, c_random_0),
        [[0.09794746, 0.00294135, 0.0032342],
         [0.00294135, 0.03595385, -0.01797943],
         [0.0032342, -0.01797943, 0.06876356]]
    )

    np.testing.assert_array_almost_equal(
        compose_covariances(t_rotate_revert, c_random_0, t_shift_complex, c_random_1),
        [[0.04409118, 0.0232232, -0.00522953],
         [0.0232232, 0.08959673, 0.01850227],
         [-0.00522953, 0.01850227, 0.06876356]]
    )

    np.testing.assert_array_almost_equal(
        compose_covariances(t_rotate_revert, c_random_1, t_shift_complex, c_random_0),
        [[0.1233138, 0.04171748, 0.07399309],
         [0.04171748, 0.08828534, 0.01587951],
         [0.07399309, 0.01587951, 0.06876356]]
    )

    for c in [c_random_0, c_random_1]:
        # here we check that identity of covariance group is 0 matrix (as opposed to eye for matrices)
        np.testing.assert_array_almost_equal(
            compose_covariances(np.zeros((3,)), np.zeros((3, 3)), t_shift_complex, c),
            c
        )
        np.testing.assert_array_almost_equal(
            compose_covariances(t_shift_complex, c, np.zeros((3,)), np.zeros((3, 3))),
            c
        )


def test_add_uncertain_transforms():
    c_random_0 = np.array(
        [[0.01432543, 0.00728222, -0.01038915],
         [0.00728222, 0.06027395, 0.00447672],
         [-0.01038915, 0.00447672, 0.01287054]])
    c_random_1 = np.array(
        [[0.03767351, 0.01022357, 0.00771092],
         [0.01022357, 0.02162842, -0.00759028],
         [0.00771092, -0.00759028, 0.05589302]])

    t_rotate_revert = np.array([0., 0., np.pi])
    t_shift_complex = np.array([-0.5, 1., -np.pi/5])

    transform, covariance = add_uncertain_transforms((t_rotate_revert, c_random_0), (t_shift_complex, c_random_1))

    np.testing.assert_array_almost_equal(
        transform,
        [0.5, -1., 2.51327412]
    )

    np.testing.assert_array_almost_equal(
        covariance,
        [[0.04409118, 0.0232232, -0.00522953],
         [0.0232232, 0.08959673, 0.01850227],
         [-0.00522953, 0.01850227, 0.06876356]]
    )


def test_compose_covariances_batch():
    c_random_0 = np.array(
        [[0.01432543, 0.00728222, -0.01038915],
         [0.00728222, 0.06027395, 0.00447672],
         [-0.01038915, 0.00447672, 0.01287054]])
    c_random_1 = np.array(
        [[0.03767351, 0.01022357, 0.00771092],
         [0.01022357, 0.02162842, -0.00759028],
         [0.00771092, -0.00759028, 0.05589302]])
    t0 = np.array([0., 0., 0.])
    t_rotate = np.array([0., 0., np.pi / 2.])
    t_rotate_revert = np.array([0., 0., np.pi])
    t_shift_complex = np.array([-0.5, 1., -np.pi/5])

    transforms0 = np.array([t0, t_rotate, t_rotate_revert, t_rotate_revert])
    covariances0 = np.array([c_random_0, c_random_1, c_random_0, c_random_1])

    transforms1 = np.array([t0, t0, t_shift_complex, t_shift_complex])
    covariances1 = np.array([c_random_1, c_random_0, c_random_1, c_random_0])

    np.testing.assert_array_almost_equal(
        compose_covariances_batch(transforms0, covariances0, transforms1, covariances1),
        [c_random_0 + c_random_1,
         [[0.09794746, 0.00294135, 0.0032342],
          [0.00294135, 0.03595385, -0.01797943],
          [0.0032342, -0.01797943, 0.06876356]],
         [[0.04409118, 0.0232232, -0.00522953],
          [0.0232232, 0.08959673, 0.01850227],
          [-0.00522953, 0.01850227, 0.06876356]],
         [[0.1233138, 0.04171748, 0.07399309],
          [0.04171748, 0.08828534, 0.01587951],
          [0.07399309, 0.01587951, 0.06876356]]
         ]
    )


def test_compose_covariances_monte_carlo():
    """
    Run monte carlo simulation to verify the correctness of compose_covariances

    Note that this only tests composing transforms w/o rotational components.

    With rotations the results don't match as well, probably due to the linearization used in the analytical
    solution.
    """

    samples = 10000000
    rng = np.random.RandomState(0)

    T1 = np.array([rng.randn(1)[0], rng.randn(1)[0], 0.0])
    T2 = np.array([rng.randn(1)[0], rng.randn(1)[0], 0.0])
    c1 = np.zeros((3, 3))
    c1[:2, :2] = np.cov(np.dot(rng.randn(4).reshape((2, 2)),
                               rng.randn(samples * 2).reshape(2, samples)))
    c2 = np.zeros((3, 3))
    c2[:2, :2] = np.cov(np.dot(rng.randn(4).reshape((2, 2)),
                               rng.randn(samples * 2).reshape(2, samples)))
    part1 = rng.multivariate_normal([0, 0], c1[:2, :2], samples)
    part2 = rng.multivariate_normal([0, 0], c2[:2, :2], samples) + part1

    sample_cov = np.cov(part2.T, bias=True)
    derived_cov = compose_covariances(T1, c1, T2, c2)

    np.testing.assert_array_almost_equal(sample_cov, derived_cov[:2, :2], decimal=2)


def test_transform_pose():
    np.testing.assert_array_almost_equal(
        transform_pose(np.array([1., 2., 3.]), np.array([0., 0., 0.])),
        [1., 2., 3.]
    )

    # translation then rotation along pose direction
    np.testing.assert_array_almost_equal(
        transform_pose(np.array([0., 0., np.pi / 2.]), np.array([1., 0., 0.])),
        [0, 1., np.pi / 2.]
    )

    np.testing.assert_array_almost_equal(
        transform_pose(np.array([1., 1., np.pi / 2.]), np.array([-1., -1., -np.pi / 2.])),
        [2, 0, 0.]
    )

    np.testing.assert_array_almost_equal(
        transform_pose(np.array([1., 1., np.pi / 2.]), np.array([-1., -1., -np.pi / 2.]), np.array([-1., -1., -np.pi / 2.])),
        [1, -1, -np.pi / 2.]
    )

    # test example in documentation of add_transforms
    left = (np.random.random_sample(3)-0.5)*10
    right = (np.random.random_sample(3)-0.5)*10
    pose = (np.random.random_sample(3)-0.5)*10
    np.testing.assert_array_almost_equal(
        transform_pose(
            transform_pose(pose, left),
            right),
        transform_pose(pose, add_transforms(left, right))
    )


def test_compose_covariances_monte_carlo_2():
    samples = 10000
    initial_pose = np.array([0., 0., 0.])

    # real data transforms and covariances
    t1 = np.array([0.55128607, 0.01298104, 0.0577444*4])
    t2 = np.array([0.52527936, -0.01775129, -0.05225523*20])
    t3 = np.array([0.2, 0.00274819, 0.00966528*10])

    c1 = np.array(
        [[1.38821517e-03, 7.95399835e-05, 0.00000000e+00],
         [7.95399835e-05, 6.21816790e-05, 6.14300601e-05],
         [0.00000000e+00, 6.14300601e-05, 1.21616496e-04]])

    c1 *= 10

    c2 = np.array(
        [[1.32319841e-03, 6.85902629e-05, 0.00000000e+00],
         [6.85902629e-05, 5.45301376e-05, 5.30262679e-05],
         [0.00000000e+00, 5.30262679e-05, 1.11086681e-04]])

    c2 *= 5

    c3 = np.array(
        [[1.37032962e-03, 1.36998417e-05, 0.00000000e+00],
         [1.36998417e-05, 2.99084369e-05, 1.13345373e-05*5],
         [0.00000000e+00, 1.13345373e-05*5, 3.08315542e-05]]
    )

    c3 *= 10

    rng = np.random.RandomState(1589478687)

    t1_samples = rng.multivariate_normal(t1, c1, samples)
    t2_samples = rng.multivariate_normal(t2, c2, samples)
    t3_samples = rng.multivariate_normal(t3, c3, samples)

    first_pose_samples = np.array([transform_pose(initial_pose, ts) for ts in t1_samples])

    true_combined_second_pose_samples = np.array([transform_pose(p, ts) for ts, p in zip(t2_samples, first_pose_samples)])
    true_combined_third_pose_samples = np.array([transform_pose(p, ts) for ts, p in zip(t3_samples, true_combined_second_pose_samples)])

    combined_transform = add_transforms(add_transforms(t1, t2), t3)

    combined_covariance = compose_covariances(
        add_transforms(t1, t2), compose_covariances(t1, c1, t2, c2),
        t3, c3)

    combined_transform_samples = rng.multivariate_normal(combined_transform, combined_covariance, samples)

    empirical_true_covariance = np.cov(true_combined_third_pose_samples, rowvar=0, bias=True)
    empirical_combined_covariance = np.cov(combined_transform_samples, rowvar=0, bias=True)

    # here we compute how well can we estimate covariance from samples
    sampling_norm = np.linalg.norm(empirical_combined_covariance - combined_covariance)
    # here we compute how well estimated norm matches compose_covariances result
    covariance_norm = np.linalg.norm(empirical_true_covariance - combined_covariance)

    # sanity check
    sampling_norm_threshold = 1e-3
    assert sampling_norm < sampling_norm_threshold
    # upper bound assertion on estimate (double of sampling norm threshold)
    assert covariance_norm < 2*sampling_norm_threshold

    # assert precise covariance to detect algorithm changes
    np.testing.assert_array_almost_equal(
        combined_covariance,
        [[0.02676195, -0.00408598, 0.00048045],
         [-0.00408598, 0.01003737, 0.00213504],
         [0.00048045, 0.00213504, 0.00207991]])


def test_add_transforms():
    np.testing.assert_array_almost_equal(
        add_transforms(np.array([0., 0., 0.]), np.array([0., 0., 0.])),
        [0., 0., 0.]
    )

    np.testing.assert_array_almost_equal(
        add_transforms(np.array([1., 2., 0.]), np.array([3., 4., 0.])),
        [4., 6., 0.]
    )

    np.testing.assert_array_almost_equal(
        add_transforms(np.array([3., 4., 0.]), np.array([1., 2., 0.])),
        [4., 6., 0.]
    )

    np.testing.assert_array_almost_equal(
        add_transforms(np.array([0., 0., 1.]), np.array([0., 0., 2.])),
        [0., 0., 3.]
    )

    np.testing.assert_array_almost_equal(
        add_transforms(np.array([1., 1., np.pi / 2.]), np.array([2., 3., 0.])),
        [-2., 3., np.pi / 2.]
    )


def test_add_transforms_batch():
    np.testing.assert_array_almost_equal(
        add_transforms_batch(
            np.array([[0., 0., 0.],
                      [1., 2., 0.],
                      [3., 4., 0.],
                      [0., 0., 1.],
                      [1., 1., np.pi / 2.]]),
            np.array([[0., 0., 0.],
                      [3., 4., 0.],
                      [1., 2., 0.],
                      [0., 0., 2.],
                      [2., 3., 0.]])),
        [[0., 0., 0.],
         [4., 6., 0.],
         [4., 6., 0.],
         [0., 0., 3.],
         [-2., 3., np.pi / 2.]]
    )


def test_from_rotation_matrix():
    for _ in range(50):
        a = np.random.rand() * np.pi*2 - np.pi
        np.testing.assert_array_almost_equal(a, from_rotation_matrix(rotation_matrix(a)))


def test_inverse_as_matrix_transpose():
    for _ in range(50):
        t0 = np.random.rand(3) * np.pi - np.pi / 2.
        result = inverse_transform(t0)
        matrix = np.linalg.inv(transform_to_homogeneous_matrix(t0))
        np.testing.assert_array_almost_equal(from_homogeneous_matrix(matrix), result)


def _generate_covariance_matrix(rng=np.random):
    '''
    http://stackoverflow.com/questions/619335/a-simple-algorithm-for-generating-positive-semidefinite-matrices
    :param rng:
    '''
    a = rng.rand(3, 3)*0.1
    c = np.dot(a, a.T)
    check_valid_cov_mat(c)
    return c


def test_compose_transforms_associativity():
    '''
    Check that add_transforms is associative: (A + B) + C == A + (B + C)
    '''
    for _ in range(100):
        t1 = np.random.rand(3) * 2*np.pi - np.pi
        t2 = np.random.rand(3) * 2*np.pi - np.pi
        t3 = np.random.rand(3) * 2*np.pi - np.pi

        t12 = add_transforms(t1, t2)
        t23 = add_transforms(t2, t3)

        np.testing.assert_array_almost_equal(
            add_transforms(t12, t3),
            add_transforms(t1, t23)
        )


def test_compose_covariances_associativity():
    '''
    Check that compose_covariances is associative:  (A + B) + C == A + (B + C),
    where A, B, C == (transform, covariance) and '+' == (add_transforms, compose_covariances)

    This is all to be able to chain transforms and covariances in a kd-tree fashion on a loop path.
    In particular, we want to prove that:
    (((A + B) + C) + D) + E == ((A + B) + (C + D)) + E
    Using the associativity, we notice that:
    ((A + B) + C) + D == (A + B) + (C + D)
    and adding E doesn't change anything.
    This allows to precompute pairs of covariances and reuse them for all paths
    '''
    for _ in range(100):
        t1 = np.random.rand(3) * 2 * np.pi - np.pi
        t2 = np.random.rand(3) * 2 * np.pi - np.pi
        t3 = np.random.rand(3) * 2 * np.pi - np.pi

        c1 = _generate_covariance_matrix()
        c2 = _generate_covariance_matrix()
        c3 = _generate_covariance_matrix()

        t12 = add_transforms(t1, t2)
        t23 = add_transforms(t2, t3)

        c12 = compose_covariances(t1, c1, t2, c2)
        c23 = compose_covariances(t2, c2, t3, c3)

        np.testing.assert_array_almost_equal(
            compose_covariances(t12, c12, t3, c3),
            compose_covariances(t1, c1, t23, c23),
        )


def test_normalize_transform_angle():
    np.testing.assert_array_almost_equal(
        normalize_transform_angle(np.array([0., 1., 3.2])),
        [0., 1., 3.2-2*np.pi]
    )

    np.testing.assert_array_almost_equal(
        normalize_transform_angle(np.array([[0., 1., -3.2], [0., 1., 0.1]])),
        [[0., 1., -3.2 + 2 * np.pi], [0., 1., 0.1]]
    )


def test_cumsum_uncertain_transforms():
    covariances = np.array([np.identity(3)*0.1, np.identity(3)*0.2, np.identity(3)*0.3])
    transforms = np.array([[0., 0., np.pi], [1., 1., -np.pi / 2.], [-1., -1., 0.]])

    cumsum_transforms, cumsum_covariances = cumsum_uncertain_transforms(transforms, covariances)
    np.testing.assert_array_almost_equal(
        cumsum_transforms,
        [transforms[0], [-1, -1, np.pi / 2.], [0., -2, np.pi / 2.]]
    )
    np.testing.assert_array_almost_equal(
        cumsum_covariances,
        [covariances[0],
         [[0.4, -0.1, 0.1],
          [-0.1, 0.4, -0.1],
          [0.1, -0.1, 0.3]],
         [[1.2, 0.2, 0.4],
          [0.2, 0.8, 0.2],
          [0.4, 0.2, 0.6]]]
    )


def test_chain_transforms():
    for _ in range(100):
        t0, t1, t2 = 2*(np.random.rand(3, 3)-0.5)*np.pi

        np.testing.assert_array_almost_equal(t0, chain_transforms(t0))
        np.testing.assert_array_almost_equal(add_transforms(t0, t1), chain_transforms(t0, t1))

        np.testing.assert_array_almost_equal(
            add_transforms(
                add_transforms(
                    t0, t1
                ),
                t2
            ),
            chain_transforms(t0, t1, t2)
        )


def test_rigid_body_transforms():
    pose_0 = np.array([1., 0., 0.])
    pose_1 = np.array([1.5, 1.5, 0.1])
    pose_to_pose_transform = transform_between_poses(pose_0, pose_1)
    np.testing.assert_array_almost_equal(pose_to_pose_transform, [0.5, 1.5, 0.1])

    robot_to_lidar = np.array([0., 1, 0.15])

    lidar_pose_0 = transform_pose(pose_0, robot_to_lidar)
    np.testing.assert_array_almost_equal(lidar_pose_0, [1., 1., 0.15])

    lidar_pose_1 = transform_pose(pose_1, robot_to_lidar)
    np.testing.assert_array_almost_equal(lidar_pose_1, [1.400167, 2.495004, 0.25])

    lidar_to_lidar_transform = transform_between_poses(lidar_pose_0, lidar_pose_1)
    np.testing.assert_array_almost_equal(lidar_to_lidar_transform, [0.619084, 1.418417, 0.1])

    np.testing.assert_array_almost_equal(
        lidar_to_lidar_transform,
        rigid_body_pose_transform(pose_to_pose_transform, robot_to_lidar)
    )

    np.testing.assert_array_almost_equal(
        pose_to_pose_transform,
        rigid_body_pose_transform(lidar_to_lidar_transform, inverse_transform(robot_to_lidar))
    )

    random_laser_translation = (np.random.random_sample(2) - 0.5) * 10
    random_yaw = (np.random.random_sample() - 0.5) * 2 * np.pi
    robot_to_lidar = np.array([
        random_laser_translation[0],
        random_laser_translation[1],
        random_yaw])

    pose_to_pose_transforms = []
    lidar_to_lidar_transforms = []
    for _ in range(100):
        pose_0 = 2*(np.random.random_sample(3) - 0.5) * np.pi
        pose_1 = 2*(np.random.random_sample(3) - 0.5) * np.pi
        pose_to_pose_transform = transform_between_poses(pose_0, pose_1)

        lidar_pose_0 = transform_pose(pose_0, robot_to_lidar)
        lidar_pose_1 = transform_pose(pose_1, robot_to_lidar)
        lidar_to_lidar_transform = transform_between_poses(lidar_pose_0, lidar_pose_1)

        np.testing.assert_array_almost_equal(
            lidar_to_lidar_transform,
            rigid_body_pose_transform(pose_to_pose_transform, robot_to_lidar)
        )

        np.testing.assert_array_almost_equal(
            pose_to_pose_transform,
            rigid_body_pose_transform(lidar_to_lidar_transform, inverse_transform(robot_to_lidar))
        )

        pose_to_pose_transforms.append(pose_to_pose_transform)
        lidar_to_lidar_transforms.append(lidar_to_lidar_transform)

    pose_to_pose_transforms = np.array(pose_to_pose_transforms)
    lidar_to_lidar_transforms = np.array(lidar_to_lidar_transforms)

    np.testing.assert_array_almost_equal(
        rigid_body_pose_transforms_batch(lidar_to_lidar_transforms, inverse_transform(robot_to_lidar)),
        pose_to_pose_transforms
    )

    np.testing.assert_array_almost_equal(
        rigid_body_pose_transforms_batch(pose_to_pose_transforms, robot_to_lidar),
        lidar_to_lidar_transforms
    )


def test_rigid_body_transforms_linear_offset():
    '''
    Check that if rigid body transform is simple translation d along the x axis of the robot (e.g. lidar)
    then there is a simple expression to compute a transform:
    [x + d*(cos(theta)-1), y+d*sin(theta), theta]
    '''

    for _ in range(100):
        distance_to_lidar = 10*(np.random.random_sample()-0.5)
        robot_to_lidar = np.array([distance_to_lidar, 0, 0])
        odom_transform = (np.random.random_sample(3) - 0.5) * 10
        lidar_transform = rigid_body_pose_transform(odom_transform, robot_to_lidar)

        np.testing.assert_almost_equal(
            lidar_transform,
            [odom_transform[0] + distance_to_lidar*(np.cos(odom_transform[2])- 1),
             odom_transform[1] + distance_to_lidar* np.sin(odom_transform[2]),
             normalize_angle(odom_transform[2])]
        )


def test_project_pose_transform_pose_relations():
    '''
    Here we state a relation between 'project' and 'transform' operations.

    Transform T that goes into transform_... operations (transform_pose)
    exists independently of coordinate system. It is relative in a sense that it applies deltas to elements (e.g. poses)
    T is the same in every coordinate system. Elements carry its coordinates with it.

    Transform Tp that goes into project_.. operations (project_poses) denotes how coordinate frames
    changed and how would elements look in the new coordinate frame.
    If T is transform between elements (poses), Tp is transform between coordinate frames.

    If coordinate frame is centered at the pose0, then result of projecting pose1 into this frame
    is equivalent to relative transform between them
    '''
    for _ in range(100):
        pose_0 = 2*(np.random.random_sample(3) - 0.5) * np.pi
        pose_1 = 2*(np.random.random_sample(3) - 0.5) * np.pi
        pose_to_pose_transform = transform_between_poses(pose_0, pose_1)
        random_transform = 2*(np.random.random_sample(3) - 0.5) * np.pi

        pose_0_in_different_coordinate_frame = project_pose(random_transform, pose_0)
        pose_1_in_different_coordinate_frame = project_pose(random_transform, pose_1)
        pose_to_pose_transform_in_different_coordinate_frame = transform_between_poses(
            pose_0_in_different_coordinate_frame, pose_1_in_different_coordinate_frame)

        # transform between poses doesn't change in a different coordinate frames
        np.testing.assert_array_almost_equal(
            pose_to_pose_transform,
            pose_to_pose_transform_in_different_coordinate_frame
        )

        # result of projecting pose1 into frame of 0 is equivalent to relative transform between them
        pose_1_in_coordinates_of_0 = project_pose(inverse_transform(pose_0), pose_1)
        np.testing.assert_array_almost_equal(
            pose_1_in_coordinates_of_0,
            pose_to_pose_transform)


def test_transformation_between_coordinate_frames():
    for _ in range(100):
        pose_0 = 2*(np.random.random_sample(3) - 0.5) * np.pi
        pose_1 = 2*(np.random.random_sample(3) - 0.5) * np.pi
        # Imagine pose0 and 1 are the same object in different coordinate frames
        # Lets find a transform between them:
        transform_between_frames = transformation_between_coordinate_frames(pose_0, pose_1)
        np.testing.assert_array_almost_equal(
            project_pose(transform_between_frames, pose_1),
            pose_0)

        # other way around
        inverse_transform_between_frames = transformation_between_coordinate_frames(pose_1, pose_0)
        np.testing.assert_array_almost_equal(
            project_pose(inverse_transform_between_frames, pose_0),
            pose_1)


def test_transforms_between_pose_pairs():
    for _ in range(10):
        src_poses = 2*(np.random.random_sample((12, 3)) - 0.5) * np.pi
        dst_poses = 2*(np.random.random_sample((12, 3)) - 0.5) * np.pi

        expected_transforms = []
        for src, dst in zip(src_poses, dst_poses):
            expected_transforms.append(transform_between_poses(src, dst))

        transforms = transforms_between_pose_pairs(src_poses, dst_poses)

        np.testing.assert_array_equal(transforms, expected_transforms)


def _reference_project_poses(transform, poses):
    """
    Python implementation of project_poses for the reference
    """
    if len(poses) == 0:
        return poses
    ph = homogenize(poses[:, :2])
    ph_moved = np.dot(transform_to_homogeneous_matrix(transform), ph.T).T
    ph_moved[:, 2] = normalize_angle(poses[:, 2] + transform[2])
    return ph_moved


def test_fast_project_poses():
    n_samples = 1000
    transforms = np.random.rand(n_samples, 3)*10. - 5.
    all_poses = []
    for _ in range(n_samples):
        all_poses.append(np.random.rand(np.random.randint(0, 1000), 3) * 1000. - 500.)
    for transform, poses in zip(transforms, all_poses):
        transform = np.ascontiguousarray(transform)
        projected = project_poses(transform, poses)
        reference = _reference_project_poses(transform, poses)
        np.testing.assert_almost_equal(projected, reference)


def test_fast_project_poses_wraparound():
    '''
    Here we show that the old method may return -np.pi instead of np.pi for an angle
    even if transform is 0. This is a change in behavior that leads to subtle differences
    in integration code
    '''
    poses = np.array([[0., 0., np.pi]])
    transform = np.array((0., 0., 0.))
    result = project_poses(transform, poses)
    reference_result = _reference_project_poses(transform, poses)

    # unfortunately numpy and cpp implementations float implementation is a bit different which leads to wrap around
    assert abs(result[0, 2] - np.pi) < 1e-10 or abs(result[0, 2] + np.pi) < 1e-10
    np.testing.assert_almost_equal(reference_result[0, 2], -np.pi)


def test_project_poses_with_time_raises():
    with pytest.raises(AssertionError):
        project_poses(np.array([0., 0., 0.]), np.array([[0., 0., 0., 0.]]))

    with pytest.raises(AssertionError):
        project_poses(np.array([0., 0., 0., 0.]), np.array([[0., 0., 0.]]))

    with pytest.raises(AssertionError):
        project_poses(np.array([0., 0., 0., 0.]), np.array([[0., 0., 0., 0.]]))


def test_transform_poses():
    original_poses = np.array([[0., 0., 0.], [1., 2., 0.]])
    np.testing.assert_array_almost_equal(
        project_poses(
            inverse_transform(np.array([1., 2., np.pi / 2.])),
            original_poses),
        np.array([[-2, 1., -np.pi / 2.], [0., 0., -np.pi / 2.]])
    )

    # check no data corruption
    np.testing.assert_array_almost_equal(
        original_poses,
        np.array([[0., 0., 0.], [1., 2., 0.]])
    )


def test_from_global_to_egocentric():
    np.testing.assert_array_almost_equal(
        from_global_to_egocentric(
            np.array([[0., 0., 0.]]),
            np.array([0., 0., 0])),
        np.array([
            [0., 0., 0.]])
    )

    np.testing.assert_array_almost_equal(
        from_global_to_egocentric(
            np.array([[0., 0., 0.]]),
            np.array([1., 2., 0])),
        np.array([
            [-1., -2., 0.]])
    )

    np.testing.assert_array_almost_equal(
        from_global_to_egocentric(
            np.array([[0., 0., 0.]]),
            np.array([1., 2., np.pi / 2.])),
        np.array([
            [-2., 1., -np.pi / 2.]])
    )

    np.testing.assert_array_almost_equal(
        from_global_to_egocentric(
            np.array([[1., 1., 0.],
                      [2., 1., np.pi/4.]]),
            np.array([-1, -1, -np.pi/4])),
        np.array([
            [0., np.sqrt(2)*2, np.pi/4],
            [np.sqrt(2)/2, np.sqrt(2)*2.5, np.pi / 2.]])
    )


def test_from_egocentric_to_global():
    np.testing.assert_array_almost_equal(
        from_egocentric_to_global(
            np.array([[0., 0., 0.]]),
            np.array([0., 0., 0])),
        np.array([
            [0., 0., 0.]])
    )

    np.testing.assert_array_almost_equal(
        from_egocentric_to_global(
            np.array([[0., 0., 0.]]),
            np.array([1., 2., 0])),
        np.array([
            [1., 2., 0.]])
    )

    np.testing.assert_array_almost_equal(
        from_egocentric_to_global(
            np.array([[1., 1., 0.],
                      [2., 1., np.pi/4.]]),
            np.array([-1, -1, -np.pi/4])),
        np.array([
            [np.sqrt(2)-1, -1, -np.pi/4],
            [1.5*np.sqrt(2)-1, -1-0.5*np.sqrt(2), 0]])
    )


def test_map_to_odom():
    np.testing.assert_array_almost_equal(
        get_map_to_odom_tranformation(map_pose=np.array([0., 0., 0.]), odom_pose=np.array([0., 0., 0.])),
        [0., 0., 0.])
    np.testing.assert_array_almost_equal(
        from_map_to_odom(np.array([[0., 0., 0.], [1., -2., np.pi/3]]),
                         get_map_to_odom_tranformation(map_pose=np.array([0., 0., 0.]), odom_pose=np.array([0., 0., 0.]))),
        [[0., 0., 0.], [1., -2., np.pi/3]]
    )

    # pure translations
    np.testing.assert_array_almost_equal(
        from_map_to_odom(np.array([[0., 0., 0.], [1., -2., np.pi/3]]),
                         get_map_to_odom_tranformation(map_pose=np.array([1., -1., 0.]), odom_pose=np.array([0., 0., 0.]))),
        [[-1., 1., 0.], [0., -1., np.pi/3]]
    )

    np.testing.assert_array_almost_equal(
        from_map_to_odom(np.array([[0., 0., 0.], [1., -2., np.pi/3]]),
                         get_map_to_odom_tranformation(map_pose=np.array([0., 0., 0.]), odom_pose=np.array([-3., 2., 0.]))),
        [[-3., 2., 0.], [-2., 0., np.pi/3]]
    )

    # translations in map and odom
    np.testing.assert_array_almost_equal(
        from_map_to_odom(np.array([[0., 0., 0.], [1., -2., np.pi/3]]),
                         get_map_to_odom_tranformation(map_pose=np.array([1.+789, -1-67., 0.]), odom_pose=np.array([0.+789, 0.+67, 0.]))),
        [[-1., 1.+67*2, 0.], [0., -1.+67*2, np.pi/3]]
    )

    # pure rotations
    np.testing.assert_array_almost_equal(
        from_map_to_odom(np.array([[0., 0., 0.], [1., -2., np.pi/3]]),
                         get_map_to_odom_tranformation(map_pose=np.array([0, 0., -np.pi / 2.]), odom_pose=np.array([0., 0., 0.]))),
        [[0., 0., np.pi / 2.], [2, 1., np.pi / 2. + np.pi/3]]
    )

    np.testing.assert_array_almost_equal(
        from_map_to_odom(np.array([[0., 0., 0.], [1., -2., np.pi / 3.]]),
                         get_map_to_odom_tranformation(map_pose=np.array([0, 0., 0.]), odom_pose=np.array([0., 0., -np.pi/3]))),
        [[0., 0., -np.pi / 3.], [-1.23205081, -1.8660254, 0.]]
    )

    # rotations in map and odom
    np.testing.assert_array_almost_equal(
        from_map_to_odom(np.array([[0., 0., 0.], [1., -2., np.pi / 3.]]),
                         get_map_to_odom_tranformation(map_pose=np.array([0, 0., -np.pi / 2.]), odom_pose=np.array([0., 0., -np.pi/3-5*np.pi / 2.]))),
        [[0., 0., -np.pi / 3.], [-1.23205081, -1.8660254, 0.]]
    )

    # translation + rotation
    np.testing.assert_array_almost_equal(
        from_map_to_odom(np.array([[0., 0., 0.], [1., -2., np.pi / 3.]]),
                         get_map_to_odom_tranformation(map_pose=np.array([-1., 1., -np.pi / 2.]), odom_pose=np.array([0., 0., 0.]))),
        [[1., 1., np.pi / 2.], [3, 2., np.pi / 2. + np.pi / 3.]]
    )

    np.testing.assert_array_almost_equal(
        from_map_to_odom(np.array([[0., 0., 0.], [1., -2., np.pi / 3.]]),
                         get_map_to_odom_tranformation(map_pose=np.array([-1., 1., -np.pi / 2.]), odom_pose=np.array([2., 2., np.pi / 3.]))),
        [[1.6339746, 3.3660254, np.pi / 2.+np.pi / 3.], [1.76794919, 5.59807621, -np.pi / 2. - np.pi / 3.]]
    )


def test_shift_and_turn():
    np.testing.assert_array_almost_equal(
        transform_poses_batch(
            np.array([[0., 0., 0.]]),
            np.array([0., 0., 0])),
        np.array([
            [0., 0., 0.]])
    )

    np.testing.assert_array_almost_equal(
        transform_poses_batch(
            np.array([[0., 0., 0.]]),
            np.array([1.1, -0.2, -np.pi / 2.])),
        np.array([[1.1, -0.2, -np.pi / 2.]])
    )

    np.testing.assert_array_almost_equal(
        transform_poses_batch(
            np.array([[1., 1., np.pi / 2.], [1., -1., -np.pi / 2.]]),
            np.array([-1., -2., 0])),
        np.array([
            [3, 0., np.pi / 2.], [-1, 0., -np.pi / 2.]])
    )

    np.testing.assert_array_almost_equal(
        transform_poses_batch(
            np.array([[0., 0., 0.]]),
            np.array([0., 0., 0.2])),
        np.array([
            [0., 0., 0.2]])
    )

    np.testing.assert_array_almost_equal(
        transform_poses_batch(
            np.array([[1., 1., np.pi / 2.], [1., -1., -np.pi / 2.]]),
            np.array([-1., -2., np.pi / 2.])),
        np.array([
            [3, 0., -np.pi], [-1, 0., 0.]])
    )


def test_interpolate_angles():
    np.testing.assert_almost_equal(interpolate_angles(0., 0.), 0)

    for i in range(20):
        try:
            np.testing.assert_almost_equal(interpolate_angles(0+i*np.pi/5, 0.2+i*np.pi/5),
                                           normalize_angle(0.1+i*np.pi/5))
        except Exception as ex:
            raise Exception("For i=%d" % i + ": " + str(ex))

    np.testing.assert_almost_equal(interpolate_angles(-0.2, 0.2), 0.)

    np.testing.assert_almost_equal(interpolate_angles(np.pi, np.pi), -np.pi)
    np.testing.assert_almost_equal(interpolate_angles(np.pi, -np.pi), -np.pi)

    np.testing.assert_almost_equal(interpolate_angles(2*np.pi+0.1, -20*np.pi-0.1), 0)
    np.testing.assert_almost_equal(interpolate_angles(0.1, -np.pi), np.pi / 2. + 0.05)
    np.testing.assert_almost_equal(interpolate_angles(0.1, np.pi+0.1), np.pi / 2. +0.1)
    np.testing.assert_almost_equal(interpolate_angles(0.1, -np.pi+0.1), np.pi / 2. +0.1)


def test_world_to_pixel():
    np.testing.assert_array_equal(
        world_to_pixel(np.array([0., 0.]), np.array([0., 0.]), 1.),
        np.array([0, 0])
    )

    np.testing.assert_array_equal(
        world_to_pixel(np.array([0., 0.]), np.array([1., 1.]), 1.),
        np.array([-1, -1])
    )

    np.testing.assert_array_equal(
        world_to_pixel(np.array([1., 1.]), np.array([0., 0.]), 1.),
        np.array([1, 1])
    )

    np.testing.assert_array_equal(
        world_to_pixel(np.array([-1., 0.]), np.array([0., 1.]), 1.),
        np.array([-1, -1])
    )

    np.testing.assert_array_equal(
        world_to_pixel(np.array([-1., 0.]), np.array([0., 1.]), 0.05),
        np.array([-20, -20])
    )

    np.testing.assert_array_equal(
        world_to_pixel(np.array([[-1., 0.]]), np.array([0., 1.]), 0.05),
        np.array([[-20, -20]])
    )

    np.testing.assert_array_equal(
        world_to_pixel(np.array([[-1., 0.],
                                 [-1., 10.],
                                 [3., -7.]]), np.array([0., 1.]), 0.05),
        np.array([[-20, -20], [-20, 180], [60, -160]])
    )

    np.testing.assert_array_equal(
        world_to_pixel(np.array([[0, 4.188]]), np.array([0., 0.]), 0.03),
        np.array([[0, 140]])
    )

    array_3d = np.array(
        [[[-1., 0.],
          [3., -7.]],
         [[-1., 0.],
          [-1., 10.]]])
    np.testing.assert_array_equal(
        world_to_pixel(array_3d.reshape(-1, 2), np.array([0., 1.]), 0.05).reshape(array_3d.shape),
        np.array([[[-20, -20], [60, -160]],
                  [[-20, -20], [-20, 180]]])
    )

    # numpy round rounds to nearest even(!) value, cpp to just neareast
    # https://stackoverflow.com/questions/50374779/how-to-avoid-incorrect-rounding-with-numpy-round
    np.testing.assert_array_equal(
        world_to_pixel(np.array([0.5, -0.5]), np.array([0., 0.]), 1.),
        np.array([0, 0])
    )
    np.testing.assert_array_equal(
        world_to_pixel(np.array([1.5, -1.5]), np.array([0., 0.]), 1.),
        np.array([2, -2])
    )

    np.testing.assert_array_equal(
        world_to_pixel(np.array([0., 0.], dtype=np.float32), np.array([0., 0.]), 1.),
        np.array([0, 0])
    )

    # this edgecase handles differences between dividing by resolution vs multiplying on 1./resolution
    np.testing.assert_array_equal(
        world_to_pixel(np.array([0., 1.075]), np.array([0., 0.]), 0.05),
        np.array([0, 22])  # becomes 21 if division is used
    )
    np.testing.assert_array_equal(
        world_to_pixel(np.array([0., 4.275]), np.array([0., 0.]), 0.03),
        np.array([0, 143])
    )
    np.testing.assert_array_equal(
        world_to_pixel(np.array([0, 2.775]), np.array([0, 0.]), 0.05),
        [0, 56]  # becomes 55  if division is used
    )

    # world to pixel works with 2d float numpy only
    with pytest.raises((AssertionError, TypeError)):
        world_to_pixel([-1., 0.], np.array([0., 1.]), 0.05)

    with pytest.raises((AssertionError, TypeError)):
        world_to_pixel(np.array([0., 1.]), [-1., 0.], 0.05)

    with pytest.raises((AssertionError, ValueError)):
        world_to_pixel(np.array([0., 1., 5]), np.array([-1., 0.]), 0.05)

    with pytest.raises((AssertionError, ValueError)):
        world_to_pixel(np.array([[-1., 0., 5],
                                 [-1., 10., 3],
                                 [3., -7., 5]]), np.array([0., 1.]), 0.05)

    # 1d origin - dangerous bug
    with pytest.raises((AssertionError, ValueError)):
        world_to_pixel(np.array([-1., 0.]), np.array([0.]), 0.05)

    # to be compatible with opencv, dtype of result has to be int, not long
    result = world_to_pixel(np.array([0., 0.]), np.array([0., 0.]), 1.)
    assert result.dtype.num == 7


if __name__ == "__main__":
    test_trivial_transform_composition()
    test_transform_composition_1()
    test_transform_composition_2()
    test_perfect_transform_loop()
    test_inverse_transform()
    test_inverse_transform_batch()
    test_inverse_transform_consistency()
    test_homogenize()
    test_de_homogenize()
    test_homogenize_de_homogenize()
    test_compose_matrix()
    test_homogeneous_matrix()
    test_project_points()
    test_project_single_point()
    test_project_no_points()
    test_transform_pose()
    test_transforms_between_poses()
    test_project_points_and_transform_relations()
    test_project_points_and_transform_relations_2()
    test_compose_identity_covariances()
    test_compose_covariances()
    test_compose_covariances_batch()
    test_add_uncertain_transforms()
    test_compose_covariances_monte_carlo()
    test_compose_covariances_monte_carlo_2()
    test_add_transforms()
    test_add_transforms_batch()
    test_rotation_matrix()
    test_from_homogeneous_matrix()
    test_transform_as_matrix_composition()
    test_inverse_as_matrix_transpose()
    test_from_rotation_matrix()
    test_compose_transforms_associativity()
    test_compose_covariances_associativity()
    test_normalize_transform_angle()
    test_cumsum_uncertain_transforms()
    test_chain_transforms()
    test_rigid_body_transforms()
    test_rigid_body_transforms_linear_offset()
    test_project_pose_transform_pose_relations()
    test_transformation_between_coordinate_frames()
    test_transforms_between_pose_pairs()
    test_correct_dehomogenize()
    test_fast_project_poses()
    test_project_poses_with_time_raises()
    test_fast_project_poses_wraparound()
    test_normalize_angles()
    test_angle_normalize_pybind_checks()
    test_from_egocentric_to_global()
    test_from_global_to_egocentric()
    test_map_to_odom()
    test_transform_poses()
    test_shift_and_turn()
    test_interpolate_angles()
    test_world_to_pixel()
