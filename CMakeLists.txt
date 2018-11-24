cmake_minimum_required(VERSION 3.7)
project(tracking_people)

set(CMAKE_CXX_STANDARD 11)
set(OPENCV_LIBRARIES opencv_highgui opencv_core opencv_imgproc opencv_video)

include_directories("/usr/local/include")
link_directories("/usr/local/lib")

set(SOURCE_FILES main.cpp utils.cpp utils.h preprocessing.cpp preprocessing.h Human.cpp Human.h)

add_executable(tracking_people ${SOURCE_FILES})
add_executable(ground_truth ground_truth_visualizer.cpp Human.cpp Human.h utils.cpp utils.h)

target_link_libraries(tracking_people ${OPENCV_LIBRARIES})
target_link_libraries(ground_truth ${OPENCV_LIBRARIES})