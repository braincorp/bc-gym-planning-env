#include "pixel_raytrace_module.h"
#include <pybind11/numpy.h>

#include <safe_array.h>
#include <iostream>
#include <cmath>
#include <limits>
#include <set>
#include <cmath>

#define PIXEL_DEBUG false

class PixelRaytraceWrapper
{
 public:
  PixelRaytraceWrapper() {
  }

  void clear_pixel(short x, short y, bool swap_xy,
                   int beam_width,
                   pybind11::safe_array_mut<uint8_t, 2>& costmap) {
    //unswap (in reverse)
    if (swap_xy) std::swap(x, y);

#if PIXEL_DEBUG
    std::cerr << "Clear pixel! (x, y)=(" << x << "," << y << ")" << std::endl;
#endif

    if (beam_width == 0) {
      costmap(y, x) = 0;
    }
    else {
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
  }


  bool bresenham_2d_precomputed(short x0, short y0, float r,
                                pybind11::safe_array<int16_t , 2>& line_defs, int index,
                                pybind11::safe_array_mut<uint8_t, 2>& costmap,
                                float resolution,
                                bool do_clearing, int beam_width, short& mark_x, short& mark_y) {
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

    //direction along each axis
    short step_x = std::max(short(-1), std::min(short(1), line_defs(index, 2)));
    short step_y = std::max(short(-1), std::min(short(1), line_defs(index, 3)));

    bool swap_xy = (dy > dx);
    if (swap_xy) {
      std::swap(x0, y0);
      std::swap(dx, dy);
      std::swap(step_x, step_y);
      std::swap(x_max, y_max);
    }

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

#if PIXEL_DEBUG
    std::cerr << "true dx " << true_dx << " remove_point " << remove_point << " r_short " << r_short
            << " r_medium " << r_medium << " r_long " << r_long << " r " << r
            << " err_short " << err_short << " err_medium " << err_medium << " err_long " << err_long << std::endl;
#endif

    // Finally, the raytracing itself
    short x_end = x0 + step_x * true_dx;
    //drift controls when to step in the short direction
    //starting value is centered
    short drift_xy  = (dx / 2) + dy;  // + dy to ensure the fist step is always in the longest direction

    short x = x0;
    short y = y0;
    // TO DO: if only marking, skip the loop (set x, y, z to their end values directly)

#if PIXEL_DEBUG
    std::cerr << "x0 " << x0 << " x_end " << x_end << " step_x " << step_x << " x_bound " << x_bound << std::endl;
#endif

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
        clear_pixel(x, y, swap_xy, beam_width, costmap);

        //update progress in other planes
        drift_xy -= dy;

        //step in short axis
        if (drift_xy < 0) {
          y += step_y;
          if (y == y_bound) break;
          drift_xy += dx;
          clear_pixel(x, y, swap_xy, beam_width, costmap);
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
        if (do_clearing && x == x_end) clear_pixel(x, y, swap_xy, beam_width, costmap);  // for consistency, always clear the last point
        if (swap_xy) std::swap(x, y);
        mark_x = x;
        mark_y = y;
        return true;
      }
    }
    return false;
  }

  pybind11::safe_array_mut<int16_t , 2> raytrace_2d_default_beam(pybind11::safe_array<uint8_t, 2>& costmap,
                                                             pybind11::safe_array<int16_t, 2>& line_defs,
                                                             pybind11::safe_array<int32_t, 1>& idx,
                                                             pybind11::safe_array<float, 1>& ranges,
                                                             pybind11::safe_array<int16_t, 1>& pixel_origin,
                                                             float resolution,
                                                             float raytrace_range, float obstacle_range,
                                                             bool do_clearing, bool do_marking, uint8_t mark_value){
    /* A wrapper for the same function with default beam width */
    return raytrace_2d(costmap, line_defs, idx, ranges, pixel_origin, resolution,
                       raytrace_range, obstacle_range, do_clearing, do_marking, mark_value);
  }

  pybind11::safe_array_mut<int16_t, 2> raytrace_2d(pybind11::safe_array<uint8_t, 2>& costmap,
                                               pybind11::safe_array<int16_t, 2>& line_defs,
                                               pybind11::safe_array<int32_t, 1>& idx,
                                               pybind11::safe_array<float, 1>& ranges,
                                               pybind11::safe_array<int16_t, 1>& pixel_origin,
                                               float resolution,
                                               float raytrace_range, float obstacle_range,
                                               bool do_clearing, bool do_marking, uint8_t mark_value,
                                               int beam_width=0){
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
    pybind11::safe_array_mut<::uint8_t, 2>& writeable_costmap = *const_cast<pybind11::safe_array_mut<uint8_t, 2>* > (&costmap);

    short max_x = costmap.shape()[1];
    short max_y = costmap.shape()[0];
    short mark_x, mark_y;
    for (int i=0; i < idx.size(); i++) {
      if (ranges[i] <= 0.) continue;
      int id = idx[i];
      short line_start_x = pixel_origin[0] + line_defs(id, 0);
      short line_start_y = pixel_origin[1] + line_defs(id, 1);
      if (line_start_x < 0 || line_start_x >= max_x || line_start_y < 0 || line_start_y >= max_y) continue;
      float range = ranges[i];
      bool valid_mark = bresenham_2d_precomputed(line_start_x, line_start_y,
                                                 std::min(range, raytrace_range), line_defs, id, writeable_costmap,
                                                 resolution, do_clearing, beam_width, mark_x, mark_y);
      if (valid_mark && range <= obstacle_range) {
        markings.push_back(mark_y);
        markings.push_back(mark_x);
#if PIXEL_DEBUG
        std::cerr << "Mark! (x, y)=(" << mark_x << "," << mark_y << ")" << std::endl;
#endif
      }
    }

    pybind11::safe_array_mut<int16_t, 2> numpy_markings({ (int)markings.size() / 2, 2 });
    int j = 0;
    for (uint i = 0; i < markings.size() / 2; ++i) {
      if (do_marking) {
        writeable_costmap(markings[j], markings[j+1]) = mark_value;
      }
      numpy_markings(i, 0) = markings[j++];
      numpy_markings(i, 1) = markings[j++];
    }
    return numpy_markings;
  }


  inline bool is_on_border(int x, int y, int x_max_ind, int y_max_ind){
    return x == 0 || y == 0 || x == x_max_ind || y == y_max_ind;
  }



  bool raytrace_2d_all_blindspot(pybind11::safe_array<uint8_t, 2>& input_map,
                                 pybind11::safe_array<uint8_t, 2>& shadow_map,
                                 pybind11::safe_array<uint8_t, 2>& blindspot_map,
                                 pybind11::safe_array<int16_t, 1>& origin,
                                 pybind11::safe_array<int16_t, 1>& closest_blindspot,
                                 const uint8_t obstacle_value,
                                 int min_cluster_size,
                                 bool return_once_marked,
                                 double max_distance_in_pixels,
                                 double negative_angle_min,
                                 double negative_angle_max,
                                 double positive_angle_min,
                                 double positive_angle_max,
                                 double angle_increment
  ){
    /**
    * This function marks all the blind spots on the blindspot_map.
    * A ray is traced between origin and the endpoints (endpoints a calculated between negative_angle_min and
    * negative_angle_max, and positive_angle_min and positive_angle_max, respectively, and with a maximum distance
    * of max_distance_in_pixels). The function return true if a blindspot has been found, false otherwise.
    * If the ray hits a shadow before it hits an obstacle,
    * it means that the visited pixel is a blindspot.
    * Parameters:
    * input_map: a Y x X uint8 matrix  which contains the current costmap with the location of obstacles
    * shadow_map: a Y x X uint8 matrix which contrains the shadows pixels
    * blindspot_map: a Y x X uint8 empty matrix where the blind spots will be marked.  It is passed as
    * const because boost numpy doesn't allow anything else, but still it is modified in place by using const_cast
    * origin: a 1d-vector of int16 representing the origin position of the raytracing
    * closest_blindspot: a 1d-vector of int16 representing the coordinates in pixel of the closest blindspot
    * obstacle_value: Value of an obstacle in the input map (usually 254)
    * min_cluster_size: The minimum of shadows to be found before a blindspot is marked on the map
    * return_once_marked: Define if the function should stop and return after a blinspot has been marked on the
        blindspot map
    * max_distance_in_pixels: Maximum distance at which we should raytrace for shadows (in pixels)
    * negative_angle_min: Min angle at which a shadow should be traced (in radians) on the negative side
    * negative_angle_max: Max angle at which a shadow should be traced (in radians) on the negative side
    * positive_angle_min: Min angle at which a shadow should be traced (in radians) on the positive side
    * positive_angle_max: Max angle at which a shadow should be traced (in radians) on the positive side
    * angle_increment: Increment in radians betweeen each ray
    * obstacle_value: Value of an obstacle in the input map (usually 254)
    */

    pybind11::safe_array_mut<int16_t, 1> endpoint({ 2 });
    pybind11::safe_array_mut<int16_t, 1> closest_found_candidate({ 2 });
    double min_distance = std::numeric_limits<double>::infinity();

    pybind11::safe_array_mut<int16_t, 1>& writeable_closest_blindspot = *const_cast<pybind11::safe_array_mut<int16_t, 1>* > (&closest_blindspot);

    double current_positive_angle = positive_angle_min;
    double current_negative_angle = negative_angle_min;
    double tmp_dist;

    while (current_positive_angle < positive_angle_max){
      // We assume that both positive side and negative side are symmetric relative to the current orientation of the robot
      // So we only check one side (here positive side), but we will apply operations on both positive and negative angle
      // For each candidate, we raytrace the blindspot. If there is a blindspot (i.e. if the closest_found_candidate is not
      // the origin point), then we check (and update closest_blindspot) if the blindspot candidate is closer than the
      // closest blindspot found so far.
      endpoint[0] = origin[0] + int(max_distance_in_pixels * cos(current_positive_angle));
      endpoint[1] = origin[1] + int(max_distance_in_pixels * sin(current_positive_angle));
      closest_found_candidate = raytrace_2d_blindspot(input_map, shadow_map, blindspot_map, origin, endpoint, obstacle_value, min_cluster_size, return_once_marked);

      tmp_dist = pow((closest_found_candidate[0]-origin[0]), 2) + pow((closest_found_candidate[0]-origin[0]), 2);
      if (tmp_dist > 0 && tmp_dist < min_distance)
      {
        writeable_closest_blindspot[0] = closest_found_candidate[0];
        writeable_closest_blindspot[1] = closest_found_candidate[1];
        min_distance = tmp_dist;
      }
      endpoint[0] = origin[0] + int(max_distance_in_pixels * cos(current_negative_angle));
      endpoint[1] = origin[1] + int(max_distance_in_pixels * sin(current_negative_angle));
      closest_found_candidate = raytrace_2d_blindspot(input_map, shadow_map, blindspot_map, origin, endpoint, obstacle_value, min_cluster_size, return_once_marked);
      tmp_dist = pow((closest_found_candidate[0]-origin[0]), 2) + pow((closest_found_candidate[0]-origin[0]), 2);
      if (tmp_dist > 0 && tmp_dist < min_distance)
      {
        writeable_closest_blindspot[0] = closest_found_candidate[0];
        writeable_closest_blindspot[1] = closest_found_candidate[1];
        min_distance = tmp_dist;
      }
      current_positive_angle += angle_increment;
      current_negative_angle -= angle_increment;
    }
    current_positive_angle = positive_angle_max;
    current_negative_angle = negative_angle_max;
    endpoint[0] = origin[0] + int(max_distance_in_pixels * cos(current_positive_angle));
    endpoint[1] = origin[1] + int(max_distance_in_pixels * sin(current_positive_angle));
    closest_found_candidate = raytrace_2d_blindspot(input_map, shadow_map, blindspot_map, origin, endpoint, obstacle_value, min_cluster_size, return_once_marked);
    tmp_dist = pow((closest_found_candidate[0]-origin[0]), 2) + pow((closest_found_candidate[0]-origin[0]), 2);
    if (tmp_dist > 0 && tmp_dist < min_distance)
    {
      writeable_closest_blindspot[0] = closest_found_candidate[0];
      writeable_closest_blindspot[1] = closest_found_candidate[1];
      min_distance = tmp_dist;
    }
    endpoint[0] = origin[0] + int(max_distance_in_pixels * cos(current_negative_angle));
    endpoint[1] = origin[1] + int(max_distance_in_pixels * sin(current_negative_angle));
    closest_found_candidate = raytrace_2d_blindspot(input_map, shadow_map, blindspot_map, origin, endpoint, obstacle_value, min_cluster_size, return_once_marked);
    tmp_dist = pow((closest_found_candidate[0]-origin[0]), 2) + pow((closest_found_candidate[0]-origin[0]), 2);
    if (tmp_dist > 0 && tmp_dist < min_distance)
    {
      writeable_closest_blindspot[0] = closest_found_candidate[0];
      writeable_closest_blindspot[1] = closest_found_candidate[1];
      min_distance = tmp_dist;
    }
    if (min_distance < std::numeric_limits<double>::infinity()){
      return true;
    }
    return false;
  }


  pybind11::safe_array_mut<int16_t, 1>
  raytrace_2d_blindspot(pybind11::safe_array<uint8_t, 2>& input_map,
                        pybind11::safe_array<uint8_t, 2>& shadow_map,
                        pybind11::safe_array<uint8_t, 2>& blindspot_map,
                        pybind11::safe_array<int16_t, 1>& origin,
                        pybind11::safe_array<int16_t, 1>& endpoint,
                        const uint8_t obstacle_value,
                        int min_cluster_size,
                        bool return_once_marked){
    /**
    * This function marks the blind spots on the blindspot_map between the origin and endpoint.
    * A ray is traced between origin and end point. If the ray hits a shadow before it hits an obstacle,
    * it means that the visited pixel is a blindspot.* Parameters:
    * input_map: a Y x X uint8 matrix  which contains the current costmap with the location of obstacles
    * shadow_map: a Y x X uint8 matrix which contrains the shadows pixels
    * blindspot_map: a Y x X uint8 empty matrix where the blind spots will be marked.  It is passed as
    * const because boost numpy doesn't allow anything else, but still it is modified in place by using const_cast
    * origin: a 1d-vector of int16 representing the origin position of the raytracing
    * endpoint: a 1d-vector of int16 representing the end position of the raytracing
    * obstacle_value: Value of an obstacle in the input map (usually 254)
    * min_cluster_size: The minimum of shadows to be found before a blindspot is marked on the map
    * return_once_marked: Define if the function should stop and return after a blinspot has been marked on the
        blindspot map
    */

    pybind11::safe_array_mut<uint8_t, 2>& writeable_blindspot_map = *const_cast<pybind11::safe_array_mut<uint8_t, 2>* > (&blindspot_map);

    pybind11::safe_array_mut<int16_t, 1> closest_blindspot({ 2 });
    closest_blindspot[0] = origin[0];
    closest_blindspot[1] = origin[1];

    int dx = abs(endpoint[0] - origin[0]);
    int dy = abs(endpoint[1] - origin[1]);
    int x = origin[0];
    int y = origin[1];
    int n = 1 + dx + dy;
    int x_inc = (endpoint[0] > origin[0]) ? 1 : -1;
    int y_inc = (endpoint[1] > origin[1]) ? 1 : -1;
    int error = dx - dy;
    int current_cluster_size = 0;
    dx *= 2;
    dy *= 2;
    bool closest_found = false;

    for (; n > 0; --n){
      if (x < 0 || x > input_map.shape()[1] - 1 || y < 0 || y > input_map.shape()[0] - 1){
        return closest_blindspot;
      }
      if (input_map(y, x) == obstacle_value){
        return closest_blindspot;
      }

      if (shadow_map(y, x) > 0){
        ++current_cluster_size;
        if (current_cluster_size >= min_cluster_size){
          writeable_blindspot_map(y, x) = 255;

          if (!closest_found){
            closest_found = true;
            closest_blindspot[0] = x;
            closest_blindspot[1] = y;
          }

          if (return_once_marked){
            return closest_blindspot;
          }
        }
      }
      if (error > 0){
        x += x_inc;
        error -= dy;
      }
      else{
        y += y_inc;
        error += dx;
      }
    }
    return closest_blindspot;

  }


  pybind11::safe_array_mut<int16_t, 2> pixel_lidar(pybind11::safe_array<uint8_t, 2>& input_map,
                                                   pybind11::safe_array<int16_t, 2>& line_defs,
                                                   pybind11::safe_array<int32_t, 1>& idx,
                                                   short origin_x, short origin_y,
                                                   const uint8_t obstacle_value){
    /* This function traces lines to find obstacles in a map. Parameters:

    input_map: a uint8 matrix with the Y x X pixel map where obstacles are to be detected
    line_defs: a m x 4 matrix of pre-calculated (dx_origin, dy_origin, dx_end, dy_end) values
        defining the start and end of each ray wrt the origin point
    idx: a n-vector with the indices in the first dimension of the lines (and lengths) array
        of the rays that will be traced.
    origin_x, origin_y: pixel origin of the given scan in the map
    obstacle_value: Value of an obstacle in the input map (usually 254)
    Returns:
    A n x 2 array with the (x, y) pixel coordinates in the pixel map of obstacles found by
        each ray; if a ray doesn't encounter and obstacle, (-1, -1) will be the value for that ray.
        Some points may be repeated if different rays find the same pixel in their way.
    */
    pybind11::safe_array_mut<int16_t, 2> numpy_obstacles({ (int)idx.size(), 2 });

    for (int i=0; i < idx.size(); i++) {
      int id = idx[i];
      short start_x = origin_x + line_defs(id, 0);
      short start_y = origin_y + line_defs(id, 1);
      short end_x = origin_x + line_defs(id, 2);
      short end_y = origin_y + line_defs(id, 3);
      bool obstacle_found = raytrace_find_first_obstacle_impl(input_map, start_x, start_y,
                                                              end_x, end_y, obstacle_value);
      if (obstacle_found) {
        numpy_obstacles(i, 0) = end_x;
        numpy_obstacles(i, 1) = end_y;
      }
      else {
        numpy_obstacles(i, 0) = -1;
        numpy_obstacles(i, 1) = -1;
      }
    }
    return numpy_obstacles;
  }


  bool raytrace_find_first_obstacle(pybind11::safe_array<uint8_t, 2>& input_map,
                                    short origin_x, short origin_y,
                                    pybind11::safe_array<int16_t, 1>& end_pose,
                                    const uint8_t obstacle_value){
    /*  Same as raytrace_find_first_obstacle but end point is passed as numpy array,
        to be usable from python */
    pybind11::safe_array_mut<int16_t, 1>& writeable_end_pose = *const_cast<pybind11::safe_array_mut<int16_t, 1>* > (&end_pose);
    short end_x = end_pose[0];
    short end_y = end_pose[1];
    bool retvalue = raytrace_find_first_obstacle_impl(input_map, origin_x, origin_y, end_x, end_y, obstacle_value);
    writeable_end_pose[0] = end_x;
    writeable_end_pose[1] = end_y;
    return retvalue;
  }

  bool raytrace_find_first_obstacle_impl(pybind11::safe_array<uint8_t, 2>& input_map,
                                         short origin_x, short origin_y,
                                         short& end_x, short& end_y,
                                         const uint8_t obstacle_value){
    /**
    * This function traces a ray between origin and endpoint on the input map.
    * If an obstacle is found in the way, it modifies the endpoint to be the point of collision
    * Parameters:
    * input_map: a Y x X uint8 matrix  which contains the current costmap with the location of obstacles
    * origin_x, origin_y: origin position of the ray tracing
    * endpoint_x, endpoint_y: end position of the ray tracing; they will be modified if an obstacle is encountered
    * obstacle_value: Value of an obstacle in the input map (usually 254)
    * Returns True if an obstacle has been encountered between origin and endpoint, else returns False
    */

    int dx = abs(end_x - origin_x);
    int dy = abs(end_y - origin_y);
    short x = origin_x;
    short y = origin_y;
    int n = 1 + dx + dy;
    short x_inc = (end_x > origin_x) ? 1 : -1;
    short y_inc = (end_y > origin_y) ? 1 : -1;
    int error = dx - dy;
    dx *= 2;
    dy *= 2;

    for (; n > 0; --n){
      if (x < 0 || x > input_map.shape()[1] - 1 || y < 0 || y > input_map.shape()[0] - 1){
        return false;
      }
      if (input_map(y, x) == obstacle_value){
        end_x = x;
        end_y = y;
        return true;
      }
      if (error > 0){
        x += x_inc;
        error -= dy;
      }
      else{
        y += y_inc;
        error += dx;
      }
    }
    return false;
  }

  void raytrace_2d_all_shadows(pybind11::safe_array<uint8_t, 2>& input_map,
                               pybind11::safe_array<uint8_t, 2>& output_map,
                               pybind11::safe_array<int16_t, 1>& origin,
                               double max_distance_in_pixels,
                               double min_angle,
                               double max_angle,
                               double angle_increment,
                               const uint8_t obstacle_value){

    /**
    * This function marks all the shadow points on the output map between the origin and endpoint points
    * with a miximum distance of max_distance_in_pixels, and in between min_angle and max_angle.
    * All the pixels that appear behind an obstacle
    * are marked as shadow (value 100)
    * Parameters:
    * input_map: a Y x X uint8 matrix  which contains the current costmap with the location of obstacles
    * output_map: a Y x X uint8 matrix which contrains the shadows pixels. It is passed as
    * const because boost numpy doesn't allow anything else, but still it is modified in place by using const_cast
    * origin: a 1d-vector of int16 representing the origin position of the ray tracing
    * max_distance_in_pixels: Maximum distance at which we should raytrace for shadows (in pixels)
    * min_angle: Min angle at which a shadow should be traced (in radians)
    * max_angle: Max angle at which a shadow should be traced (in radians)
    * angle_increment: Increment in radians betweeen each ray
    * obstacle_value: Value of an obstacle in the input map (usually 254)
    */
    pybind11::safe_array_mut<int16_t, 1> endpoint({ 2 });

    double current_angle = min_angle;
    while (current_angle < max_angle){
      // We loop between min angle and max angle at angle increment
      endpoint[0] = origin[0] + int(max_distance_in_pixels * cos(current_angle));
      endpoint[1] = origin[1] + int(max_distance_in_pixels * sin(current_angle));
      raytrace_2d_shadows(input_map, output_map, origin, endpoint, obstacle_value);
      current_angle += angle_increment;
    }
    // We process the last angle
    current_angle = max_angle;
    endpoint[0] = origin[0] + int(max_distance_in_pixels * cos(current_angle));
    endpoint[1] = origin[1] + int(max_distance_in_pixels * sin(current_angle));
    raytrace_2d_shadows(input_map, output_map, origin, endpoint, obstacle_value);
  }

  void raytrace_2d_shadows(pybind11::safe_array<uint8_t, 2>& input_map,
                           pybind11::safe_array<uint8_t, 2>& output_map,
                           pybind11::safe_array<int16_t, 1>& origin,
                           pybind11::safe_array<int16_t, 1>& endpoint,
                           const uint8_t obstacle_value){
    /**
    * This function marks the shadow points on the output map between the origin and endpoint points.
    *  A ray is traced between origin and end point. All the pixels that appear behind an obstacle
    * are marked as shadow (value 100)
    * Parameters:
    * input_map: a Y x X uint8 matrix  which contains the current costmap with the location of obstacles
    * output_map: a Y x X uint8 matrix which contrains the shadows pixels. It is passed as
    * const because boost numpy doesn't allow anything else, but still it is modified in place by using const_cast
    * origin: a 1d-vector of int16 representing the origin position of the ray tracing
    * endpoint: a 1d-vector of int16 representing the end position of the ray tracing
    * obstacle_value: Value of an obstacle in the input map (usually 254)
    */

    pybind11::safe_array_mut<uint8_t, 2>& writeable_output_map = *const_cast<pybind11::safe_array_mut<uint8_t, 2>* > (&output_map);

    int dx = abs(endpoint[0] - origin[0]);
    int dy = abs(endpoint[1] - origin[1]);
    int x = origin[0];
    int y = origin[1];
    int n = 1 + dx + dy;
    int x_inc = (endpoint[0] > origin[0]) ? 1 : -1;
    int y_inc = (endpoint[1] > origin[1]) ? 1 : -1;
    int error = dx - dy;
    dx *= 2;
    dy *= 2;

    bool mark_shadow = false;

    for (; n > 0; --n){
      if (x < 0 || x > input_map.shape()[1] - 1 || y < 0 || y > input_map.shape()[0] - 1){
        return;
      }
      if (mark_shadow){
        writeable_output_map(y, x) = 100;
      }
      else if (input_map(y, x) == obstacle_value){
        mark_shadow = true;
      }
      if (error > 0){
        x += x_inc;
        error -= dy;
      }
      else{
        y += y_inc;
        error += dx;
      }
    }
  }


  void raytrace_clean_and_occupied(pybind11::safe_array<uint16_t, 2>& clear_counts_map,
                                   pybind11::safe_array<int16_t, 1>& origin,
                                   pybind11::safe_array<int16_t, 2>& endpoints){
    /**
    * This function increments the empty cells between origin and endpoint in clear_counts_map.
    *
    * Parameters:
    * clear_counts_map: a Y x X uint16 matrix which contains the current clear_counts_map
    * origin: a 1d-vector of int16 representing the origin position of the ray tracing
    * endpoints: an N x 2 array of int16 representing the end position of the ray tracing
    */

    pybind11::safe_array_mut<uint16_t, 2>& writeable_clear_counts_map = *const_cast<pybind11::safe_array_mut<uint16_t, 2>* > (&clear_counts_map);

    std::set<std::pair<int,int> > empty_pixels;

    // Use set to find all touched pixels and then write once.
    for (uint i = 0; i < endpoints.shape()[0]; i++)
    {
      int dx = abs(endpoints(i, 0) - origin[0]);
      int dy = abs(endpoints(i, 1) - origin[1]);
      int x = origin[0];
      int y = origin[1];
      int n = 1 + dx + dy;
      int x_inc = (endpoints(i, 0) > origin[0]) ? 1 : -1;
      int y_inc = (endpoints(i, 1) > origin[1]) ? 1 : -1;
      int error = dx - dy;
      dx *= 2;
      dy *= 2;

      for (; n > 0; --n){
        if (x < 0 || x > clear_counts_map.shape()[1] - 1 || y < 0 || y > clear_counts_map.shape()[0] - 1){
          break;
        }

        std::pair<int, int> empty_pixel(y, x);

        if (error > 0){
          x += x_inc;
          error -= dy;
        }
        else{
          y += y_inc;
          error += dx;
        }

        if (empty_pixel.second == endpoints(i, 0) && empty_pixel.first == endpoints(i, 1)){
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

  void raytrace_cliff(pybind11::safe_array<uint8_t, 2>& costmap,
                      pybind11::safe_array<int16_t, 2>& origins,
                      pybind11::safe_array<int16_t, 2>& endpoints,
                      pybind11::safe_array<uint8_t, 1>& marking_weights,
                      uint8_t marking_threshold){
    /**
    * This function increments the traced lines (defined by (x, y) starting and ending points,
        one for each line) by the corresponding marking weight in the given costmap. Used
        for marking cliff areas.
    *
    * Parameters:
    * costmap: a Y x X uint8 matrix with mark values
    * origins: a n x 2  array of (x, y) line origins for ray tracing
    * endpoints: an n x 2 array of (x, y) line ends for ray tracing
    * marking_weights: a n-length vector with the marking weight of each line, or a 1-element
        vector with the common weight of all lines
    * marking_threshold: maximum value of marks in the costmap (other than 255 for NO_INFORMATION)
    */

    pybind11::safe_array_mut<uint8_t, 2>& writeable_costmap = *const_cast<pybind11::safe_array_mut<uint8_t, 2>* > (&costmap);
    uint n_lines = origins.shape()[0];
    std::set<std::tuple<int, int, uint8_t> > marked_pixels;  // use a set to remove duplicates and avoid incrementing twice the same pixel

    for (uint i = 0; i < n_lines; ++i) {
      int x = origins(i, 0);
      int y = origins(i, 1);
      int dx = abs(endpoints(i, 0) - x);
      int dy = abs(endpoints(i, 1) - y);
      int n = 1 + dx + dy;
      int x_inc = (endpoints(i, 0) > x) ? 1 : -1;
      int y_inc = (endpoints(i, 1) > y) ? 1 : -1;
      int error = dx - dy;
      int weight_index = i;
      if (marking_weights.shape()[0] == 1) weight_index = 0;
      dx *= 2;
      dy *= 2;

      for (; n > 0; --n) {
        if (x < 0 || x >= costmap.shape()[1] || y < 0 || y >= costmap.shape()[0])
          break;
        std::tuple<int, int, uint8_t> pixel(y, x, marking_weights[weight_index]);
        marked_pixels.insert(pixel);

        if (error > 0) {
          x += x_inc;
          error -= dy;
        }
        else {
          y += y_inc;
          error += dx;
        }
      }
    }

    // Mark the unique pixels
    std::set<std::tuple<int, int, uint8_t> >::iterator it;
    for (it = marked_pixels.begin(); it != marked_pixels.end(); it++) {
      uint8_t current_weight = std::get<2>(*it);
      uint8_t current_counts = writeable_costmap(std::get<0>(*it), std::get<1>(*it));
      if (current_counts == 255) {
        writeable_costmap(std::get<0>(*it), std::get<1>(*it)) = current_weight;
      }
      else if (current_counts >= marking_threshold - current_weight) {
        writeable_costmap(std::get<0>(*it), std::get<1>(*it)) = marking_threshold;
      }
      else writeable_costmap(std::get<0>(*it), std::get<1>(*it)) += current_weight;
    }
  }

  pybind11::safe_array_mut<int16_t, 2> raytrace_2d_glass(
    pybind11::safe_array<uint8_t, 2>& costmap,
    pybind11::safe_array<int16_t, 2>& line_defs,
    pybind11::safe_array<int32_t, 1>& idx,
    pybind11::safe_array<float, 1>& ranges,
    pybind11::safe_array<int16_t, 1>& pixel_origin,
    float resolution,
    float raytrace_range,
    float obstacle_range,
    bool do_clearing,
    bool do_marking,
    uint8_t mark_value,
    int beam_width=0){
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
    A l x 3 array with the (y, x) pixel coordinates of marks and the corresponding indices
    (corresponding to input arrays such as line_defs, idx and ranges); l <= n because
        some of the rays will not produce marks (because they fall outside the map, or further than
        obstacle_range or raytrace_range). The marks are returned independently of whether do_marking
        is True or False
    */

    std::vector<int16_t> markings;
    pybind11::safe_array_mut<uint8_t, 2>& writeable_costmap = *const_cast<pybind11::safe_array_mut<uint8_t, 2>* > (&costmap);

    short max_x = costmap.shape()[1];
    short max_y = costmap.shape()[0];
    short mark_x, mark_y;
    for (int i=0; i < idx.size(); i++) {
      if (ranges[i] <= 0.) continue;
      int id = idx[i];
      short line_start_x = pixel_origin[0] + line_defs(id, 0);
      short line_start_y = pixel_origin[1] + line_defs(id, 1);
      if (line_start_x < 0 || line_start_x >= max_x || line_start_y < 0 || line_start_y >= max_y) continue;
      float range = ranges[i];
      bool valid_mark = bresenham_2d_precomputed(line_start_x, line_start_y,
                                                 std::min(range, raytrace_range), line_defs, id, writeable_costmap,
                                                 resolution, do_clearing, beam_width, mark_x, mark_y);
      if (valid_mark && range <= obstacle_range) {
        markings.push_back(mark_y);
        markings.push_back(mark_x);
        markings.push_back(i);
#if PIXEL_DEBUG
        std::cout << "Mark! (x, y)=(" << mark_x << "," << mark_y << ")" << std::endl;
#endif
      }
    }

    pybind11::safe_array_mut<int16_t, 2> numpy_markings({ (int)markings.size() / 3, 3 });
    int j = 0;
    for (uint i = 0; i < markings.size() / 3; ++i) {
      if (do_marking) {
        writeable_costmap(markings[j], markings[j+1]) = mark_value;
      }
      numpy_markings(i, 0) = markings[j++];
      numpy_markings(i, 1) = markings[j++];
      numpy_markings(i, 2) = markings[j++];
    }
    return numpy_markings;
  }
};

void registerPixelRaytraceModule(pybind11::module module)
{
  pybind11::class_<PixelRaytraceWrapper>(module, "PixelRaytraceWrapper")
    .def(pybind11::init<>())
    .def("raytrace_2d", &PixelRaytraceWrapper::raytrace_2d)
    .def("raytrace_2d", &PixelRaytraceWrapper::raytrace_2d_default_beam)
    .def("raytrace_2d_shadows", &PixelRaytraceWrapper::raytrace_2d_shadows)
    .def("raytrace_2d_all_shadows", &PixelRaytraceWrapper::raytrace_2d_all_shadows)
    .def("raytrace_2d_blindspot", &PixelRaytraceWrapper::raytrace_2d_blindspot)
    .def("raytrace_2d_all_blindspot", &PixelRaytraceWrapper::raytrace_2d_all_blindspot)
    .def("raytrace_find_first_obstacle", &PixelRaytraceWrapper::raytrace_find_first_obstacle)
    .def("raytrace_clean_and_occupied", &PixelRaytraceWrapper::raytrace_clean_and_occupied)
    .def("raytrace_cliff", &PixelRaytraceWrapper::raytrace_cliff)
    .def("pixel_lidar", &PixelRaytraceWrapper::pixel_lidar)
    .def("raytrace_2d_glass", &PixelRaytraceWrapper::raytrace_2d_glass);
}
