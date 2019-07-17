#include <pybind11/pybind11.h>
#include "pixel_raytracing_utils.h"
#include "pixel_raytrace_module.h"

PYBIND11_MODULE(_bc_gym_cpp_module, m) {
  registerPixelRaytracingUtilsModule(m.def_submodule("pixel_raytracing_utils"));
  registerPixelRaytraceModule(m.def_submodule("pixel_raytrace_module"));
}
