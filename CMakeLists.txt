cmake_minimum_required(VERSION 2.8)
project(mi_proyecto_features)
# C++11 support
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
add_executable( apk odometry.cpp )
if(PCL_FOUND AND OpenCV_FOUND)
include_directories(${PCL_INCLUDE_DIRS})
include_directories( ${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
link_directories(${OpenCV_LIB_DIR})
add_definitions(${PCL_DEFINITIONS})
target_link_libraries( apk ${PCL_LIBRARIES})
target_link_libraries( apk ${OpenCV_LIBS})
endif()

