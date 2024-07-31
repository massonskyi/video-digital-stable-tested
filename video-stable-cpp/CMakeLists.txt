cmake_minimum_required(VERSION 3.10)

# Project name and version
project(VideoStabilization)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
add_executable(VideoStabilization main.cpp)

# Link the OpenCV libraries
target_link_libraries(VideoStabilization ${OpenCV_LIBS})
