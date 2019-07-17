#include <cstdio>
#include <iostream>
#include <cstdint>
#include <climits>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <list>
#include <assert.h>
#include <cstddef>
#include <array>
#include <malloc.h>
#include <range.h>
#include <safe_array.h>

namespace py = pybind11;
using namespace pybind11::literals;


void
clear_pixel_with_beam(
    short x, short y, bool swap_xy,
    int beam_width,
    py::safe_array_mut<uint8_t, 2>& costmap
    ) {
    /*
    Clear ray with non-single pixel fat beam
    :param x: x start of the ray
    :param y: y start of the ray
    :param swap_xy: whether x and y are swapped
    :param costmap: costmap to clear
    */
    //unswap (in reverse)
    if (swap_xy) std::swap(x, y);

    for (int x_beam = x - beam_width; x_beam <= x + beam_width; ++x_beam) {
        if (x_beam >= 0 && x_beam < costmap.shape()[1]) {
            for (int y_beam = y - beam_width; y_beam <= y + beam_width; ++y_beam) {
                if (y_beam >= 0 && y_beam < costmap.shape()[0]) {
                    costmap(y_beam, x_beam) = 0;
                }
            }
        }
    }
}


void
clear_ray(
    py::safe_array_mut<uint8_t, 2>& costmap,
    short drift_xy,
    short x_end,
    short &x,
    short &y,
    short dx,
    short dy,
    short step_x,
    short step_y,
    short x_bound,
    short y_bound
    ) {
    /*
    Clear ray with single pixel beam
    :param costmap: costmap to clear
    :param drift_xy:  when to step in the short direction?
    :param x_end: final x coordinate of the beam (measured obstacle)
    :param x: x start of the ray (also returned to track progress)
    :param y: y start of the ray (also returned to track progress)
    :param dx: unit length of the beam in x direction
    :param dy: unit length of the beam in y direction
    :param step_x: x direction (+1, -1) of the beam
    :param step_y: y direction (+1, -1) of the beam
    :param x_bound: corresponding x bound of the beam (-1 or shape)
    :param y_bound: corresponding y bound of the beam (-1 or shape)
    */
    for (; x != x_end; x += step_x) {
        if (x == x_bound) break;
        costmap(x, y) = 0;
        //update progress in other planes
        drift_xy -= dy;

        //step in short axis
        if (drift_xy < 0) {
            y += step_y;
            if (y == y_bound) break;
            drift_xy += dx;
            costmap(x, y) = 0;
        }
    }
}


void
clear_ray_swapped(
    py::safe_array_mut<uint8_t, 2>& costmap,
    short drift_xy,
    short x_end,
    short &x,
    short &y,
    short dx,
    short dy,
    short step_x,
    short step_y,
    short x_bound,
    short y_bound
    )
{
    /*
    Clear ray with single pixel beam, but x and y swapped.
    :param costmap: costmap to clear
    :param drift_xy:  when to step in the short direction?
    :param x_end: final x coordinate of the beam (measured obstacle)
    :param x: x start of the ray (also returned to track progress)
    :param y: y start of the ray (also returned to track progress)
    :param dx: unit length of the beam in x direction
    :param dy: unit length of the beam in y direction
    :param step_x: x direction (+1, -1) of the beam
    :param step_y: y direction (+1, -1) of the beam
    :param x_bound: corresponding x bound of the beam (-1 or shape)
    :param y_bound: corresponding y bound of the beam (-1 or shape)
    */
    for (; x != x_end; x += step_x) {
        if (x == x_bound) break;
        costmap(y, x) = 0;
        //update progress in other planes
        drift_xy -= dy;

        //step in short axis
        if (drift_xy < 0) {
            y += step_y;
            if (y == y_bound) break;
            drift_xy += dx;
            costmap(y, x) = 0;
        }
    }
}

bool
raytracing_body_impl(
    py::safe_array_mut<uint8_t, 2>& costmap,
    short x0,
    short y0,
    short dx,
    short dy,
    short step_x,
    short step_y,
    short x_max,
    short y_max,
    short x_bound,
    short y_bound,
    bool swap_xy,
    short x_end,
    short drift_xy,
    bool remove_point,
    short& mark_x,
    short& mark_y
    ) {
    /*
    Raytrace with clearing only, with single pixel wide beam
    :param costmap: costmap to clear
    :param x0: start x of the beam to raytrace
    :param y0: start y of the beam to raytrace
    :param dx: unit length of the beam in x direction
    :param dy: unit length of the beam in y direction
    :param step_x: x direction (+1, -1) of the beam
    :param step_y: y direction (+1, -1) of the beam
    :param x_max: shape of the costmap x
    :param y_max: shape of the costmap y
    :param x_bound: corresponding x bound of the beam (-1 or shape)
    :param y_bound: corresponding y bound of the beam (-1 or shape)
    :param swap_xy: whether x and y are swapped
    :param x_end: final x coordinate of the beam (measured obstacle)
    :param drift_xy: when to step in the short direction?
    :param remove_point: a flag if we need to move one pixel back to mark it
    :param mark_x: last pixel to mark (x) returned
    :param mark_y: last pixel to mark (y) returned
    :return: whether last pixel is in bounds
    */

    short x = x0;
    short y = y0;
    // TO DO: if only marking, skip the loop (set x, y, z to their end values directly)

    if (swap_xy) {
        clear_ray(
            costmap,
            drift_xy,
            x_end,
            x,
            y,
            dx,
            dy,
            step_x,
            step_y,
            x_bound,
            y_bound
        );

    } else {
        clear_ray_swapped(
            costmap,
            drift_xy,
            x_end,
            x,
            y,
            dx,
            dy,
            step_x,
            step_y,
            x_bound,
            y_bound
        );
    }

    // Final adjustments depending on the exact ending pixel candidate
    if (x != x_end) return false;  // out of bounds, no marks
    if (remove_point) {
        if (x != x0) x -= step_x;
    }
    if (x != x_bound && y != y_bound) {  // got to the desired endpoint, check termination and marking
        if (x == x_end) {
            // for consistency, always clear the last point
            if (swap_xy) costmap(x, y) = 0;
            else costmap(y, x) = 0;
        }
        if (swap_xy) std::swap(x, y);
        mark_x = x;
        mark_y = y;
        return true;
    }
    return false;
}

bool
raytracing_body_no_clearing_impl(
    py::safe_array_mut<uint8_t, 2>& costmap,
    short x0,
    short y0,
    short dx,
    short dy,
    short step_x,
    short step_y,
    short x_max,
    short y_max,
    short x_bound,
    short y_bound,
    bool swap_xy,
    short x_end,
    short true_dx,
    bool remove_point,
    short& mark_x,
    short& mark_y
    ) {
    /*
    Raytrace to find pixels to mark, without clearing (single pixel wide beam)
    :param costmap: costmap to clear
    :param x0: start x of the beam to raytrace
    :param y0: start y of the beam to raytrace
    :param dx: unit length of the beam in x direction
    :param dy: unit length of the beam in y direction
    :param step_x: x direction (+1, -1) of the beam
    :param step_y: y direction (+1, -1) of the beam
    :param x_max: shape of the costmap x
    :param y_max: shape of the costmap y
    :param x_bound: corresponding x bound of the beam (-1 or shape)
    :param y_bound: corresponding y bound of the beam (-1 or shape)
    :param swap_xy: whether x and y are swapped
    :param x_end: final x coordinate of the beam (measured obstacle)
    :param true_dx: actual length of the beam in x direction (till obstacle)
    :param remove_point: a flag if we need to move one pixel back to mark it
    :param mark_x: last pixel to mark (x) returned
    :param mark_y: last pixel to mark (y) returned
    :return: whether last pixel is in bounds
    */

    short x = x0;
    short y = y0;
    // TO DO: if only marking, skip the loop (set x, y, z to their end values directly)

    x = x_end;
    int end_drift_xy = (int) (dx / 2) - (int) (true_dx - 1) * (int) dy;
    short n_y_steps = - (end_drift_xy - dx + 1) / dx;
    y += step_y * n_y_steps;
    // Check if the loop would result in out-of-bounds
    // As a special case, if x_end == x_bound, we may have to continue,
    //    even though it is at the very bound.
    if ((x != x_bound && (x < 0 || x >= x_max)) || y < 0 || y >= y_max) return false;

    // Final adjustments depending on the exact ending pixel candidate
    if (x != x_end) return false;  // out of bounds, no marks
    if (remove_point) {
        if (x != x0) x -= step_x;
    }
    if (x != x_bound && y != y_bound) {  // got to the desired endpoint, check termination and marking
        if (swap_xy) std::swap(x, y);
        mark_x = x;
        mark_y = y;
        return true;
    }
    return false;
}


bool
raytracing_body_with_beam_impl(
    py::safe_array_mut<uint8_t, 2>& costmap,
    short x0,
    short y0,
    short dx,
    short dy,
    short step_x,
    short step_y,
    short x_max,
    short y_max,
    short x_bound,
    short y_bound,
    bool swap_xy,
    bool do_clearing,
    short x_end,
    short true_dx,
    short drift_xy,
    bool remove_point,
    int beam_width,
    short& mark_x,
    short& mark_y
    ) {
    /*
    Raytrace to clear and find pixels to mark with multipixel fat beam
    :param costmap: costmap to clear
    :param x0: start x of the beam to raytrace
    :param y0: start y of the beam to raytrace
    :param dx: unit length of the beam in x direction
    :param dy: unit length of the beam in y direction
    :param step_x: x direction (+1, -1) of the beam
    :param step_y: y direction (+1, -1) of the beam
    :param x_max: shape of the costmap x
    :param y_max: shape of the costmap y
    :param x_bound: corresponding x bound of the beam (-1 or shape)
    :param y_bound: corresponding y bound of the beam (-1 or shape)
    :param swap_xy: whether x and y are swapped
    :param do_clearing: whether perform clearing or only find the final points
    :param x_end: final x coordinate of the beam (measured obstacle)
    :param true_dx: actual length of the beam in x direction (till obstacle)
    :param drift_xy: when to step in the short direction?
    :param remove_point: a flag if we need to move one pixel back to mark it
    :param beam_width: width of the beam
    :param mark_x: last pixel to mark (x) returned
    :param mark_y: last pixel to mark (y) returned
    :return: whether last pixel is in bounds
    */

    short x = x0;
    short y = y0;
    // TO DO: if only marking, skip the loop (set x, y, z to their end values directly)

    if (!do_clearing) {  // no need to trace, skip directly to the mark
        x = x_end;
        int end_drift_xy = (int) (dx / 2) - (int) (true_dx - 1) * (int) dy;
        short n_y_steps = - (end_drift_xy - dx + 1) / dx;
        y += step_y * n_y_steps;
        drift_xy = (short) (end_drift_xy + n_y_steps * dx);
        // Check if the loop would result in out-of-bounds
        // As a special case, if x_end == x_bound, we may have to continue,
        //    even though it is at the very bound.
        if ((x != x_bound && (x < 0 || x >= x_max)) || y < 0 || y >= y_max) return false;
        }
    else {  // trace with clearing
        for (; x != x_end; x += step_x) {
            if (x == x_bound) break;
            clear_pixel_with_beam(x, y, swap_xy, beam_width, costmap);

            //update progress in other planes
            drift_xy -= dy;

            //step in short axis
            if (drift_xy < 0) {
                y += step_y;
                if (y == y_bound) break;
                drift_xy += dx;
                clear_pixel_with_beam(x, y, swap_xy, beam_width, costmap);
            }

        }
    }

    // Final adjustments depending on the exact ending pixel candidate
    if (x != x_end) return false;  // out of bounds, no marks
    if (remove_point) {
        if (x != x0) x -= step_x;
    }
    if (x != x_bound) {  // got to the desired endpoint, check termination and marking
        if (y != y_bound) {
            if (do_clearing && x == x_end) {
                // for consistency, always clear the last point
                clear_pixel_with_beam(x, y, swap_xy, beam_width, costmap);
            }
            if (swap_xy) std::swap(x, y);
            mark_x = x;
            mark_y = y;
            return true;
        }
    }
    return false;
}

bool
bresenham_2d_precomputed(
    short x0, short y0, float r,
    const py::safe_array<int16_t, 2>& line_defs,
    int index,
    py::safe_array_mut<uint8_t, 2>& costmap,
    float resolution,
    bool do_clearing,
    int beam_width,
    short& mark_x,
    short& mark_y) {
/* A 2d Bresenham raytracing algorithm given a line definition and a distance r. It clears
    the pixels in the given 2d costmap along the ray, and returns the pixel coordinates where
    the final mark would go (but it does NOT mark the given costmap).
    x0, y0: start coordinates of the line
    r: distance along the line to be traced (in world units)
    line_defs: a m x 4 matrix of pre-calculated (x0, y0) and (dx, dy) values for rays that can be traced
        dx, dy are signed.
    index: the particular ray in line_defs to be traced
    costmap: the 2d array of pixels where clearing and marking will take place
    resolution: world units per pixel
    do_clearing: whether to clear along the ray; if false, no clearing happens, and
        the function merely returns the pixel coordinates where the mark should go
    beam_width: if > 0, the clearing rays will be thickened in the x and y direction by this number
        of pixels (on each side)
    mark_x, mark_y: here the coordinates of the mark will be returned */
    short x_max = costmap.shape()[1];
    short y_max = costmap.shape()[0];

    float r2 = resolution * resolution;

    //length along each axis
    short dx = std::abs(line_defs(index, 2));
    short dy = std::abs(line_defs(index, 3));

    //direction of movement along each axis (+1 or -1)
    short step_x = std::max(short(-1), std::min(short(1), line_defs(index, 2)));
    short step_y = std::max(short(-1), std::min(short(1), line_defs(index, 3)));

    bool swap_xy = (dy > dx);
    if (swap_xy) {
        std::swap(x0, y0);
        std::swap(dx, dy);
        std::swap(step_x, step_y);
        std::swap(x_max, y_max);
    }

    // Relevant boundary of the image for each axis (shape-1 or 0)
    short x_bound = step_x > 0 ? x_max:-1;
    short y_bound = step_y > 0 ? y_max:-1;

    // The following does the main trick of this function: calculate
    // which pixel along the line corresponds to the given distance.
    // It's a bit messy because of the integer arithmetic, which
    // only allows us to find 3 possible candidates for the line end;
    // the correct among the 3 candidates has to be found by brute force
    float dx2 = (float) dx * (float) dx;
    float y_sq_term = r2 * (float) dy * (float) dy / dx2;
    float total_r_scale = sqrt(r2 + y_sq_term);
    float true_dx_short = std::floor(r / total_r_scale);
    float true_dx_long = true_dx_short + 1.0;
    // Here we compute the 3 candidate pixels for the mark
    float r_short = true_dx_short * total_r_scale;
    float r_medium = sqrt(r_short * r_short + r2 * (2 * true_dx_short + 1));
    float r_long = true_dx_long * total_r_scale;
    float err_short = std::abs(r_short - r);
    float err_medium = std::abs(r_medium - r);
    float err_long = std::abs(r_long - r);
    // And now the logic to select the closest candidate to the given distance
    bool remove_point = false;
    short true_dx = (short) (1 + true_dx_short + 0.5);
    if (err_short < err_medium) {
        remove_point = true;
    }
    else if (err_medium < err_long) {
    }
    else {
        true_dx += 1;
        remove_point = true;
    }

    // Finally, the raytracing itself
    short x_end = x0 + step_x * true_dx;
    //drift controls when to step in the short direction
    //starting value is centered
    short drift_xy  = (dx / 2) + dy;  // + dy to ensure the fist step is always in the longest direction

    if (beam_width == 0) {
        if (do_clearing) {
            return raytracing_body_impl(
                costmap,
                x0,
                y0,
                dx,
                dy,
                step_x,
                step_y,
                x_max,
                y_max,
                x_bound,
                y_bound,
                swap_xy,
                x_end,
                drift_xy,
                remove_point,
                mark_x,
                mark_y);
        } else {
            return raytracing_body_no_clearing_impl(
                costmap,
                x0,
                y0,
                dx,
                dy,
                step_x,
                step_y,
                x_max,
                y_max,
                x_bound,
                y_bound,
                swap_xy,
                x_end,
                true_dx,
                remove_point,
                mark_x,
                mark_y);
        }
    } else {
        return raytracing_body_with_beam_impl(
            costmap,
            x0,
            y0,
            dx,
            dy,
            step_x,
            step_y,
            x_max,
            y_max,
            x_bound,
            y_bound,
            swap_xy,
            do_clearing,
            x_end,
            true_dx,
            drift_xy,
            remove_point,
            beam_width,
            mark_x,
            mark_y);
    }
}


py::safe_array<int16_t, 2>
raytrace_2d(
    py::safe_array_mut<uint8_t, 2>& costmap,
    const py::safe_array<int16_t, 2>& line_defs,
    const py::safe_array<int32_t, 1>& idx,
    const py::safe_array<float, 1>& ranges,
    const py::safe_array<int16_t, 1>& pixel_origin,
    float resolution,
    float raytrace_range,
    float obstacle_range,
    bool do_clearing,
    bool do_marking,
    uint8_t mark_value,
    int beam_width) {
    /* This function traces predefined clearing scans in a 2d costmap. Parameters:

    costmap: a Y x X uint8 matrix that will be traced, It is passed as const because
        boost numpy doesn't allow anything else, but still it is modified in place by using const_cast
    line_defs: a m x 4 matrix of pre-calculated (x0, y0) and (dx, dy) values for rays that can be traced;
        dx, dy are signed
    idx: a n-vector with the indices in the first dimension of the lines (and lengths) array
        of the rays that will be traced.
    ranges: another n-vector with the range of each ray to trace (in meters); a range of 0 means the ray
        will be ignored
    pixel_origin: (x, y) origin of the given scan in the costmap
    resolution: world units per pixel
    raytrace_range: rays will only be traced up to this distance (in world units) from their origins
    obstacle_range: marks will be returned only for rays that end closer than this from their origin
    do_clearing: if True, rays are cleared
    do_marking: if True, marks are also set in the volume (and in the column_counts); marks are set
        with a 1 in the pixel
    mark_value: value for marks in the costmap
    beam_width: if > 0, the clearing rays will be thickened in the x and y direction by this number
        of pixels (on each side)
    Returns:
    A l x 2 array with the (y, x) pixel coordinates of marks; l <= n because
        some of the rays will not produce marks (because they fall outside the map, or further than
        obstacle_range or raytrace_range). The marks are returned independently of whether do_marking
        is True or False
    */
    std::vector<int16_t> markings;

    short max_x = costmap.shape()[1];
    short max_y = costmap.shape()[0];
    short mark_x, mark_y;

    for (int i=0; i < idx.size(); i++) {
        if (ranges[i] <= 0.) continue;
        int id = idx[i];
        short line_start_x = pixel_origin(0) + line_defs(id, 0);
        short line_start_y = pixel_origin(1) + line_defs(id, 1);
        if (line_start_x < 0 || line_start_x >= max_x || line_start_y < 0 || line_start_y >= max_y) continue;
        float range = ranges[i];
        bool valid_mark = bresenham_2d_precomputed(line_start_x, line_start_y,
            std::min(range, raytrace_range), line_defs, id, costmap,
            resolution, do_clearing, beam_width, mark_x, mark_y);
        if (valid_mark && range <= obstacle_range) {
            markings.push_back(mark_y);
            markings.push_back(mark_x);
            #if PIXEL_DEBUG
            std::cerr << "Mark! (x, y)=(" << mark_x << "," << mark_y << ")" << std::endl;
            #endif
        }
    }

    py::safe_array_mut<int16_t, 2> numpy_markings({ (int)markings.size() / 2, 2 });
    int j = 0;
    for (int i = 0; i < markings.size() / 2; ++i) {
        if (do_marking) {
            costmap(markings[j], markings[j+1]) = mark_value;
        }
        numpy_markings(i, 0) = markings[j++];
        numpy_markings(i, 1) = markings[j++];
    }
    return numpy_markings;
}


void raytrace_clean_on_input_map(const py::safe_array_mut<uint8_t, 2>& input_map,
                                 const py::safe_array_mut<uint16_t, 2>& clear_counts_map,
                                 const py::safe_array<int16_t, 2>& line_defs,
                                 const py::safe_array<int32_t, 1>& idx,
                                 short origin_x, short origin_y,
                                 const uint8_t obstacle_value){
    /*
    * This function increments the empty cells between [origin_x, origin_y] and the first obstacles encountered in
    cost map

    * Parameters:
    * clear_counts_map: a Y x X uint16 matrix which contains the current clear_counts_map
    * input_map: a uint8 matrix with the Y x X pixel map where obstacles are to be detected
    * line_defs: a m x 4 matrix of pre-calculated (dx_origin, dy_origin, dx_end, dy_end) values
        defining the start and end of each ray wrt the origin point
    * idx: a n-vector with the indices in the first dimension of the lines (and lengths) array
        of the rays that will be traced.
    * origin_x, origin_y: pixel origin of the given scan in the map
    * obstacle_value: Value of an obstacle in the input map (usually 254)
    */

    // the map that will be returned
    py::safe_array_mut<uint16_t, 2>& writeable_clear_counts_map = *const_cast<py::safe_array_mut<uint16_t, 2>* > (&clear_counts_map);
    std::set<std::pair<int,int> > empty_pixels;

    for (int i=0; i < idx.size(); i++) {
        int id = idx[i];
        short start_x = origin_x + line_defs(id, 0);
        short start_y = origin_y + line_defs(id, 1);
        short end_x = origin_x + line_defs(id, 2);
        short end_y = origin_y + line_defs(id, 3);
        int dx = abs(end_x - start_x);
        int dy = abs(end_y - start_y);
        short x = start_x;
        short y = start_y;
        int n = 1 + dx + dy;
        short x_inc = (end_x > start_x) ? 1 : -1;
        short y_inc = (end_y > start_y) ? 1 : -1;
        int error = dx - dy;
        dx *= 2;
        dy *= 2;

        // loop over n to find the obstacles
        for (; n > 0; --n){
            if (x < 0 || x > input_map.shape()[1] - 1 || y < 0 || y > input_map.shape()[0] - 1){
                break;
            }

            // store empty pixel
            std::pair<int, int> empty_pixel(y, x);

            if (error > 0){
                x += x_inc;
                error -= dy;
            }
            else{
                y += y_inc;
                error += dx;
            }

            // if we at the obstacles break the loop
            if (input_map(y, x) == obstacle_value){
                end_x = x;
                end_y = y;
                break; // exclude endpoint for tracing empty space
            }

            if (empty_pixel.second == end_x && empty_pixel.first == end_y){
                continue; // exclude endpoint for tracing empty space
            }

            empty_pixels.insert(empty_pixel);
        }

    }

    // Write empty pixels
    std::set<std::pair<int, int> >::iterator it;

    for (it = empty_pixels.begin(); it != empty_pixels.end(); it++) {
        uint16_t current_counts = writeable_clear_counts_map((*it).first, (*it).second);
        if (current_counts < 65535) writeable_clear_counts_map((*it).first, (*it).second) += 1;
    }
}

PYBIND11_MODULE(_pixel_raytracing_utils, m)
{
    m.def("raytrace_2d_impl",
          &raytrace_2d,
          "costmap"_a.noconvert(),
          "line_defs"_a.noconvert(),
          "idx"_a.noconvert(),
          "ranges"_a.noconvert(),
          "pixel_origin"_a.noconvert(),
          "resolution"_a,
          "raytrace_range"_a,
          "obstacle_range"_a,
          "do_clearing"_a,
          "do_marking"_a,
          "mark_value"_a,
          "beam_width"_a
    );

    m.def("raytrace_clean_on_input_map_impl",
          &raytrace_clean_on_input_map,
          "input_map"_a.noconvert(),
          "clear_counts_map"_a.noconvert(),
          "line_defs"_a.noconvert(),
          "idx"_a.noconvert(),
          "origin_x"_a,
          "origin_y"_a,
          "obstacle_value"_a
    );
}