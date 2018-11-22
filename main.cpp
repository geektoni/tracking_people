#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>

#include "utils.h"
#include "preprocessing.h"

using namespace cv;

int main(int argc, char ** argv) {

	// Parse the command line and get the
	// various arguments
	std::vector<string> arguments {"-f"};
	InputParser parser(argc, argv, arguments);
	parser.parse();

	// Open the video and check if it is correct
	// otherwise return with an error.
	VideoCapture video(parser.get("-f"));
	if (!video.isOpened())
		return -1;

	// Current frame
	Mat frame, previous, lines_mask, tracking_lines;

	// Foreground mask (generated by GMM)
	Mat fg;

	// result
	Mat fg_copy;

	// Background Remover object
	FindPeople bg_rem;

	// Contours;
	vector<vector<Point>> contours;
	vector<Rect> boundRect;

	// Frame counter
	int i=0;

	while (true) {

		// Get the frame
		video >> frame;

		// Initialize the tracking mask
		if(lines_mask.empty())
		{
			lines_mask = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC3);
		}

		// If we reach the end of the video, we exit
		if (frame.empty())
			exit(0);

		// Extract people and copy the frame (prevent further modifications)
		Mat fg = bg_rem.find_people(frame);
		fg.copyTo(fg_copy);

		// Find contours and boundin boxes
		Mat drawing = bg_rem.find_contours(fg, frame, true, contours, boundRect);

		// Get the bounding rectangle centers
		vector<Point2f> current_centers = FindPeople::compute_center(boundRect);
		vector<Point2f> next_centers;

		// The tracking is performed starting from the
		// second frame. We also do the tracking by measuring the
		// displacement each 5 frame to have less noise.
		frame.copyTo(tracking_lines);
		if (!previous.empty() && i%5==0)
		{
			 next_centers = bg_rem.track_people_optical(previous, frame, contours, boundRect);

			// Print a line between the points and the result
			for (int j = 0; j < current_centers.size(); ++j) {
				line(lines_mask, current_centers[0], next_centers[0], Scalar(255,0,0), 3);
			}

			// Merge the lines and the frame
			merge_images(tracking_lines, lines_mask);

			// Update the previous frame
			frame.copyTo(previous);
		}

		// Update for the first thame the previous frame
		if (previous.empty() && i%5==0)
			frame.copyTo(previous);

		// Print everything on screen
		//namedWindow("Threshold",WINDOW_NORMAL);
		//resizeWindow("Threshold", 600, 600);
		//imshow("Threshold", fg_copy);

		namedWindow("Detect",WINDOW_NORMAL);
		resizeWindow("Detect", 600, 600);
		imshow("Detect", drawing);

		namedWindow("Tracking",WINDOW_NORMAL);
		resizeWindow("Tracking", 600, 600);
		imshow("Tracking", tracking_lines);

		// Increment the frame counter
		i++;

		waitKey(1);
	}


	return 0;
}