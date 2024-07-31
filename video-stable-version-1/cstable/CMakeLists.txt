cmake_minimum_required(VERSION 3.10)

# Project name and version
project(VidStab)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/vidstable.cpp
    src/vidstab_utils.cpp
    src/auto_border_utils.cpp
    src/border_utils.cpp
    src/cv2_utils.cpp
    src/frame_queue.cpp
    src/frame.cpp
    src/general_utils.cpp
    src/layer_utils.cpp
    src/main_utils.cpp
)

set(INCLUDE_DIRECTORIES
    includes/vidstable.h
    includes/vidstab_utils.h
    includes/auto_border_utils.h
    includes/border_utils.h
    includes/cv2_utils.h
    includes/frame_queue.h
    includes/frame.h
    includes/general_utils.h
    includes/layer_utils.h
    includes/main_utils.h
    includes/pop_deque.h
)

# Add executable
add_executable(VidStab 
main.cpp
${INCLUDE_DIRECTORIES}
${SOURCES}

)

# Link OpenCV libraries
target_link_libraries(VidStab ${OpenCV_LIBS})
