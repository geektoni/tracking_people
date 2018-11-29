# Detecting and Tracking (People) Motion

This repository contains the work done for the course of Computer Vision and Multimedia Analysis of
the University of Trento. We were asked to devise a system to count and track moving pedestrian
in a given video acquired from top view.

## Install

This project was tested on Ubuntu 16.04 and it requires OpenCV 2.4.13 in order to operate properly.
It uses CMake to manage the default build process (CMake version must be 2.8 or greater).
The `evaluator.py` was tested with Python 3 and it requires some additional libraries: pandas, numpy,
and sklearn.

The installation procedure is the following:
```bash
cd tracking_people
mkdir build
cd build
cmake ../
make tracking_people
```
This will generate an executable called `tracking_people` inside the `build` directory.

There is also the possibility to build another target called `ground_truth` which will
display on screen the ids and trajectories of the selected pedestrians (10, 36, and 42)
which were used for testing the accuracy of the predicted trajectories.
As usual, the procedure is the following:
```bash
cd tracking_people
cd build
make ground_truth
```

## Running

To see the algorithm in action, the base procedure is the following
```bash
cd tracking_people
cd build
./tracking_people ../data/A1_test.mp4 kalman
```
If one wants to run the procedure by just using the simple region-based algorithm, then he needs to
change the algorithm value from `kalman` to `simple`.

## Documentation

It is possible also to generate the HTML code documentation. This requires Doxygen to be installed
in the system. To build the docs the procedure is the following:
```bash
cd tracking_people
cd build
make doc
```
