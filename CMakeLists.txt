cmake_minimum_required(VERSION 2.8.3)
project(bc_gym_planning_env)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bc_gym_planning_env)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -O3 -Wall -fpic")

# find current version of python 2.7
execute_process(
        COMMAND "python2.7" -c "from distutils import sysconfig; print(sysconfig.get_python_inc())"
        OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    "${PROJECT_SOURCE_DIR}/inc"
    "${PROJECT_SOURCE_DIR}/deps/pybind11/include"
)


add_library(pixel_raytracing_utils SHARED ${PROJECT_SOURCE_DIR} src/pixel_raytracing_utils.cpp)
target_link_libraries(pixel_raytracing_utils ${PYTHON_LIBRARIES})
set_target_properties(pixel_raytracing_utils PROPERTIES SUFFIX ".so" PREFIX "_")

add_library(pixel_raytrace_module SHARED ${PROJECT_SOURCE_DIR} src/pixel_raytrace_module.cpp)
target_link_libraries(pixel_raytrace_module ${PYTHON_LIBRARIES})
set_target_properties(pixel_raytrace_module PROPERTIES SUFFIX ".so" PREFIX "_")