from __future__ import print_function, absolute_import, division

from builtins import range
import numpy as np
import logging

from bc_gym_planning_env import PixelRaytraceWrapper, raytrace_2d


def test_2d_raytracing_basic():
    '''
    Testing the c++ predefined 2d raytracing
    '''

    # We create a few rays, each ray has its own x coordinate and increasing
    # length up to a maximum along the y axis
    n_rays = 20
    max_length = 15
    resolution = 1.
    costmap = np.ones((max_length, n_rays), dtype=np.uint8)
    reconstructed_costmap = costmap.copy()
    pixel_origin = np.array((0, 0), dtype=np.int16)
    line_defs = np.zeros((n_rays, 4), dtype=np.int16)
    for i in range(n_rays):
        line_defs[i, :] = [i, 0, 0, max_length]

    # Now we create a scan choosing randomly a few of the rays and assigning them fixed, increasing ranges
    scan_size = n_rays // 2
    idx = np.random.randint(0, n_rays, scan_size).astype(np.int32)
    ranges = (-1. + np.arange(scan_size)).astype(np.float32)

    # Now let's check that the module behaves as expected
    marks = raytrace_2d(costmap, line_defs, idx, ranges, pixel_origin,
                        resolution, 64., 64., True, False, 254, 0)
    marks_idx = 0
    for i in range(scan_size):
        x, y = idx[i], max_length
        r = -1 + i
        actual_y = min(y, r)
        if actual_y > 0:
            reconstructed_costmap[:actual_y + 1, x] = 0
        mark_pos = np.round(max(0, ranges[i])).astype(np.int)
        if r <= 0. or mark_pos >= max_length:
            continue  # there will be no mark if the expected mark index is beyond the length of the ray
        assert np.array_equal(marks[marks_idx], [mark_pos, x])
        marks_idx += 1
    assert np.array_equal(costmap, reconstructed_costmap)
    original_marks = marks

    # Check the cpp marking
    marks = raytrace_2d(costmap, line_defs, idx, ranges, pixel_origin,
                        resolution, 64., 64., True, True, 254, 0)
    marked_costmap = reconstructed_costmap.copy()
    marked_costmap[marks[:, 0], marks[:, 1]] = 254
    assert np.array_equal(costmap, marked_costmap)
    # Test that the only-marking case produces the same marks as the clearing + marking case
    marks_only = raytrace_2d(costmap, line_defs, idx, ranges, pixel_origin,
                             resolution, 64., 64., False, True, 254, 0)
    assert np.array_equal(marks, marks_only)
    assert np.array_equal(costmap, marked_costmap)

    # Check shifting the origin
    # Examples where either the lines are completely outside the volume or
    # (in the last case) the beginning of the lines are outside: no trace
    for origin in [(n_rays, 0), (-n_rays, 0), (0, max_length), (0, -max_length), (0, -1)]:
        pixel_origin = np.array(origin, dtype=np.int16)  # completely outside
        costmap = np.ones((max_length, n_rays), dtype=np.uint8)
        marks = raytrace_2d(costmap, line_defs, idx, ranges, pixel_origin,
                            resolution, 64., 64., True, False, 254, 0)
        assert len(marks) == 0 and np.all(costmap == 1)
    # In the next example the rays are partially outside, but the outside part
    # is toward the end of the ray, so some clearing and marking does happen
    pixel_origin = np.array((0, max_length // 3), dtype=np.int16)
    costmap = np.ones((max_length, n_rays), dtype=np.uint8)
    shifted_reconstructed_costmap = costmap.copy()
    marks = raytrace_2d(costmap, line_defs, idx, ranges, pixel_origin,
                        resolution, 64., 64., True, False, 254, 0)
    shifted_reconstructed_costmap[max_length // 3:, :] = reconstructed_costmap[:-(max_length // 3), :]
    assert np.array_equal(costmap, shifted_reconstructed_costmap)
    expected_marks = original_marks[original_marks[:, 0] < max_length - max_length // 3].copy()
    expected_marks[:, 0] += max_length // 3
    assert np.array_equal(marks, expected_marks)
    # Here some rays are fully outside, some fully inside
    pixel_origin = np.array((-1, 0), dtype=np.int16)
    costmap = np.ones((max_length, n_rays), dtype=np.uint8)
    shifted_reconstructed_costmap = costmap.copy()
    marks = raytrace_2d(costmap, line_defs, idx, ranges, pixel_origin,
                        resolution, 64., 64., True, False, 254, 0)
    shifted_reconstructed_costmap[:, :n_rays - 1] = reconstructed_costmap[:, 1:n_rays]
    assert np.array_equal(costmap, shifted_reconstructed_costmap)
    expected_marks = original_marks[original_marks[:, 1] >= 1].copy()
    expected_marks[:, 1] -= 1
    assert np.array_equal(marks, expected_marks)
    # And here all are inside, but shifted
    pixel_origin = np.array((1, 0), dtype=np.int16)
    costmap = np.ones((max_length, n_rays), dtype=np.uint8)
    shifted_reconstructed_costmap = costmap.copy()
    marks = raytrace_2d(costmap, line_defs, idx, ranges, pixel_origin,
                        resolution, 64., 64., True, False, 254, 0)
    shifted_reconstructed_costmap[:, 1:] = reconstructed_costmap[:, :n_rays - 1]
    assert np.array_equal(costmap, shifted_reconstructed_costmap)
    expected_marks = original_marks[original_marks[:, 1] < n_rays - 1].copy()
    expected_marks[:, 1] += 1
    assert np.array_equal(marks, expected_marks)
    # Finally, let's make sure negative line coordinates also work; here with all rays inside...
    shifted_lines = line_defs.copy()
    shifted_lines[:, 0] -= 2
    shifted_lines[:, 1] -= 5
    pixel_origin = np.array((2, 5), dtype=np.int16)
    costmap = np.ones((max_length, n_rays), dtype=np.uint8)
    marks = raytrace_2d(costmap, shifted_lines, idx, ranges, pixel_origin,
                        resolution, 64., 64., True, False, 254, 0)
    assert np.array_equal(costmap, reconstructed_costmap)
    assert np.array_equal(marks, original_marks)
    # ...and here with some rays outside
    shifted_lines = line_defs.copy()
    shifted_lines[:, 0] -= 3
    shifted_lines[:, 1] -= 5
    pixel_origin = np.array((2, 5), dtype=np.int16)
    costmap = np.ones((max_length, n_rays), dtype=np.uint8)
    shifted_reconstructed_costmap = costmap.copy()
    marks = raytrace_2d(costmap, shifted_lines, idx, ranges, pixel_origin,
                        resolution, 64., 64., True, False, 254, 0)
    shifted_reconstructed_costmap[:, :n_rays - 1] = reconstructed_costmap[:, 1:n_rays]
    assert np.array_equal(costmap, shifted_reconstructed_costmap)
    expected_marks = original_marks[original_marks[:, 1] >= 1].copy()
    expected_marks[:, 1] -= 1
    assert np.array_equal(marks, expected_marks)


def test_2d_raytrace_length():
    '''
    Tests accuracy of ray length in 2d precomputed raytracing
    '''
    size = np.array([140, 200], dtype=np.int)  # x, y, z volume size
    resolution = 1.

    costmap = np.ones((size[1], size[0]), dtype=np.uint8)
    pixel_origin = size[:2].astype(np.int16) // 2

    r = np.linspace(0.1, 49., 500).astype(np.float32)  # we trace with a variety of lengths along a single ray
    # Check lines along each axis, in both directions
    for sign in [-1, 1]:
        for axis in range(2):
            costmap[:] = 1
            line_def = np.zeros((1, 4), dtype = np.int16)
            line_def[0, axis + 2] = 30 * sign
            marks = raytrace_2d(costmap, line_def, np.zeros(len(r)).astype(np.int32),
                                r, pixel_origin, resolution, 64., 64., True, False, 254, 0)
            expected_marks = (np.outer(sign * np.round(r), np.identity(2)[axis]) + size // 2)[:, [1, 0]]
            assert np.array_equal(marks, expected_marks)
            # Also check that only marking produces the same marks as clearing and marking
            marks_only = raytrace_2d(costmap, line_def, np.zeros(len(r)).astype(np.int32),
                                     r, pixel_origin, resolution, 64., 64., False, True, 254, 0)
            assert np.array_equal(marks, marks_only)
            assert set([tuple(p) for p in marks]) == set([tuple(p) for p in np.vstack(np.where(costmap==254)).T])

    # Check lines at 45 degrees in 4 directions
    for sign in [[-1, -1], [-1, 1], [1, -1], [1, 1]]:
        costmap = np.ones((size[1], size[0]), dtype=np.uint8)
        line_def = np.zeros((1, 4), dtype = np.int16)
        line_def[0, 2:4] = 30 * np.array(sign)
        marks = raytrace_2d(costmap, line_def, np.zeros(len(r)).astype(np.int32),
                            r, pixel_origin, resolution, 64., 64., True, False, 254, 0)
        expected_marks = (np.outer(np.round(r / np.sqrt(2)), sign) + size // 2)[:, [1, 0]]
        assert np.amax(np.sum(np.abs(marks - expected_marks), axis=1)) <= 1  # make sure all marks are at most 1 pixel from the expected
        trace = np.vstack(np.where(costmap==0)).T
        # Also check that only marking produces the same marks as clearing and marking
        marks_only = raytrace_2d(costmap, line_def, np.zeros(len(r)).astype(np.int32),
                                 r, pixel_origin, resolution, 64., 64., False, True, 254, 0)
        assert np.array_equal(marks, marks_only)
        assert set([tuple(p) for p in marks]) == set([tuple(p) for p in np.vstack(np.where(costmap==254)).T])

        # Check that the marks correspond to the pixels along the line that have distance to origin closest to given r
        ray_dists = np.sqrt(np.sum(((trace - size[[1, 0]] // 2) * [resolution, resolution]) ** 2, axis=1))  # distance of each pixel along the trace
        closest_trace_point_idx = np.argmin(np.abs(ray_dists[:, None] - r[None, :]), axis=0)
        assert np.array_equal(marks, trace[closest_trace_point_idx])

        # Check that wide beam clearing works
        for beam_width in [1, 2, 3]:
            costmap[:] = 1
            _ = raytrace_2d(costmap, line_def, np.zeros(len(r)).astype(np.int32),
                            r, pixel_origin, resolution, 64., 64., True, False, 254, beam_width)
            wide_trace = np.vstack(np.where(costmap==0)).T

            def _array_to_set(a):
                return set(tuple(t) for t in a)

            inflated_trace = _array_to_set(trace)
            for x in range(-beam_width, beam_width + 1):
                for y in range(-beam_width, beam_width + 1):
                    inflated_trace = inflated_trace.union(_array_to_set(trace + [x, y]))
            assert _array_to_set(wide_trace) == inflated_trace

    # Finally, check the distance part in random directions
    rng = np.random.RandomState(0)
    n_lines = 10
    n_r = 500

    for direction in rng.randint(-50, 50, (n_lines, 2)):
        resolution = 0.1 + 5 * rng.rand()
        max_r = min(size * [resolution, resolution] / 2)
        r = np.linspace(0.1, max_r * 0.95, n_r).astype(np.float32)  # we trace with a variety of lengths along a single ray
        pixel_origin = size[:2].astype(np.int16) // 2
        costmap = np.ones((size[1], size[0]), dtype=np.uint8)
        line_def = np.zeros((1, 4), dtype = np.int16)
        line_def[0, 2:4] = direction
        marks = raytrace_2d(costmap, line_def, np.zeros(len(r)).astype(np.int32),
                            r, pixel_origin, resolution, max_r, max_r, True, False, 254, 0)
        trace = np.vstack(np.where(costmap==0)).T
        # Also check that only marking produces the same marks as clearing and marking
        marks_only = raytrace_2d(costmap, line_def, np.zeros(len(r)).astype(np.int32),
                                 r, pixel_origin, resolution, max_r, max_r, False, True, 254, 0)
        assert np.array_equal(marks, marks_only)
        assert set([tuple(p) for p in marks]) == set([tuple(p) for p in np.vstack(np.where(costmap==254)).T])

        # Check that the marks correspond to the pixels along the line that have distance to origin closest to given r
        ray_dists = np.sqrt(np.sum(((trace - size[[1, 0]] // 2) * [resolution, resolution]) ** 2, axis=1))  # distance of each pixel along the trace
        mark_dists = np.sqrt(np.sum(((marks - size[[1, 0]] // 2) * [resolution, resolution]) ** 2, axis=1))  # distance of each mark
        closest_trace_point_idx = np.argmin(np.abs(ray_dists[:, None] - r[None, :]), axis=0)
        bad = np.unique(np.where(marks != trace[closest_trace_point_idx])[0])
        if len(bad):
            assert np.max(np.abs(marks[bad] - trace[closest_trace_point_idx][bad])) <= 1  # wrong by at most 1 pixel
            assert np.all(np.sum(np.abs(marks[bad] - trace[closest_trace_point_idx][bad]), axis=1) <= 2)  # wrong by at most 2 coords per pixel
            # wrong by less than half a pixel per wrong dimension
            assert np.all(
                np.abs(r[bad] - mark_dists[bad]) - np.abs(r[bad] - ray_dists[closest_trace_point_idx][bad]) < resolution * np.sqrt(2))


def raytrace_2d_reference(*args, **kwargs):
    m = PixelRaytraceWrapper()
    return m.raytrace_2d(*args, **kwargs)


if __name__ == "__main__":
    logging.getLogger().setLevel(level=logging.INFO)

    test_2d_raytracing_basic()
    test_2d_raytrace_length()
    logging.info('Passed all tests!')
