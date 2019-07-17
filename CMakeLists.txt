cmake_minimum_required(VERSION 2.8.3)
project(bc_gym_planning_env)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
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

set(SOURCE_FILES
        src/pixel_raytrace_module.cpp
        src/pixel_raytracing_utils.cpp
)

add_library(bc_gym_cpp_module SHARED ${PROJECT_SOURCE_DIR} ${SOURCE_FILES})
target_link_libraries(bc_gym_cpp_module ${PYTHON_LIBRARIES})
set_target_properties(bc_gym_cpp_module PROPERTIES SUFFIX ".so" PREFIX "_")
