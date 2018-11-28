# Detecting and Tracking (People) Motion

This repository contains the work done for the course of Computer Vision and Multimedia Analysis of
the University of Trento. We were asked to devise a system to count and track moving pedestrian
in a given video acquired from top view.

## Install

This project was tested on Ubuntu 16.04 and it requires OpenCV 2.4.13 in order to operate properly.
The `evaluator.py` was tested with Python 3 and it requires some additional libraries: pandas, numpy,
and sklearn.

The installation procedure is the following:
`
cd tracking_people
mkdir build
cd build
cmake ../
make tracking_people
`
This will generate an executable called `tracking_people` inside the `build` directory.

## Run it

To see the algorithm in action, the base procedure is the following
`
cd tracking_people
cd build
./tracking_people -f ../data/A1_assignment/A1_test.mp4 -alg kalman
`
If one wants to run the procedure by just using the simple region-based version, then he needs to
change the `-alg` value from `kalman` to `simple`.


tracking_people
Tracking People with OpenCV.

The task of this assignment is to count and track people in a given video acquired from top view.

The accomplishment of the assignment will be evaluated on the two tasks separately (counting and tracking).

Counting: the student has to provide a frame-by-frame information related to the actual number of people present in the scene. As an optional task, the number of people entering/exiting a specific gate can be provided.

Tracking: for each pedestrian in the scene, provide the trajectory, as a separate output file, to be compared with the provided ground truth. As a validation metric, the average displacement error will be considered for a selected number of pedestrians.