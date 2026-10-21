# Out-of-tree build of the vendored X30 MotionSDK (sources untouched).
# Uses the venv's pip pybind11 because the SDK's pybind11 git submodule isn't vendored.
#
#   cmake -S sdk -B sdk/build -Dpybind11_DIR=$(.venv/bin/python -m pybind11 --cmakedir) \
#         -DPython_EXECUTABLE=$PWD/.venv/bin/python && cmake --build sdk/build -j
cmake_minimum_required(VERSION 3.15)
project(x30_motion_sdk_sim CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(SDK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/x30_motion_sdk)
add_compile_options(-w)

file(GLOB SDK_SRC ${SDK_DIR}/src/*.cpp)
add_library(x30_motion_sdk STATIC ${SDK_SRC})
target_include_directories(x30_motion_sdk PUBLIC ${SDK_DIR}/include)
target_link_libraries(x30_motion_sdk PUBLIC pthread m rt dl)

# DR's C++ stand-up demo, unchanged. OFF by default: its main.cpp calls SendToRobot() with the
# hard-coded default 192.168.1.103, i.e. it talks to a REAL robot on the network, never the sim.
option(BUILD_DR_EXAMPLE "Build DR's C++ example (targets the real robot at 192.168.1.103)" OFF)
if(BUILD_DR_EXAMPLE)
  add_executable(motion_sdk_example ${SDK_DIR}/example/main.cpp ${SDK_DIR}/example/motion_sdk_example.cpp)
  target_include_directories(motion_sdk_example PRIVATE ${SDK_DIR}/example)
  target_link_libraries(motion_sdk_example x30_motion_sdk)
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(deeprobotics_x30_motion_sdk_py ${SDK_DIR}/python/motion_sdk_pybind.cpp)
target_link_libraries(deeprobotics_x30_motion_sdk_py PRIVATE x30_motion_sdk)
set_target_properties(deeprobotics_x30_motion_sdk_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../lib)

# Prints the wire-struct sizes the sim bridge must match.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/sizes.cpp
"#include <cstdio>\n#include \"command.h\"\n#include \"x30_types.h\"\nusing namespace x30;\n"
"int main(){ EthCommand c{}; c.type=1; c.count=0xABCDEF; unsigned w[3]; static_assert(sizeof(c)==12,\"\");\n"
"  __builtin_memcpy(w,&c,12);\n"
"  std::printf(\"EthCommand=%zu RobotCmdSDK=%zu RobotDataSDK=%zu ImuDataSDK=%zu word2(type=1,count=0xABCDEF)=0x%08x\\n\",\n"
"    sizeof(EthCommand), sizeof(RobotCmdSDK), sizeof(RobotDataSDK), sizeof(ImuDataSDK), w[2]); }\n")
add_executable(print_sizes ${CMAKE_CURRENT_BINARY_DIR}/sizes.cpp)
target_include_directories(print_sizes PRIVATE ${SDK_DIR}/include)
