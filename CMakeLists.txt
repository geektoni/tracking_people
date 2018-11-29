cmake_minimum_required(VERSION 2.8)
project(tracking_people)

# Fix in order to make it compatible also with older version
# of CMake.
if (CMAKE_VERSION VERSION_LESS "3.1")
	if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
    endif()
  else()
    set(CMAKE_CXX_STANDARD 11)
endif()

set(OPENCV_LIBRARIES opencv_highgui opencv_core opencv_imgproc opencv_video)

include_directories("/usr/local/include")
link_directories("/usr/local/lib")

set(SOURCE_FILES main.cpp utils.cpp utils.h FindPeople.cpp FindPeople.h Human.cpp Human.h)

add_executable(tracking_people ${SOURCE_FILES})
add_executable(ground_truth ground_truth_visualizer.cpp Human.cpp Human.h utils.cpp utils.h)

target_link_libraries(tracking_people ${OPENCV_LIBRARIES})
target_link_libraries(ground_truth ${OPENCV_LIBRARIES})

find_package(Doxygen)
if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    # note the option ALL which allows to build the docs together with the application
    add_custom_target( doc ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM )
else (DOXYGEN_FOUND)
    message("[Warning] Doxygen need to be installed to generate the doxygen documentation.")
endif (DOXYGEN_FOUND)
