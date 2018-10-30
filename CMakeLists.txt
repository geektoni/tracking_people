cmake_minimum_required(VERSION 3.7)
project(tracking_people)

set(CMAKE_CXX_STANDARD 11)
set(OPENCV_LIBRARIES opencv_highgui opencv_core opencv_imgproc opencv_video)

include_directories("/usr/local/include")
link_directories("/usr/local/lib")

set(SOURCE_FILES main.cpp utils.cpp utils.h)
add_executable(tracking_people ${SOURCE_FILES})
target_link_libraries(tracking_people ${OPENCV_LIBRARIES})