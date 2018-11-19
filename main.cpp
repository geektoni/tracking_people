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
	Mat frame;

	// Foreground mask (generated by GMM)
	Mat fg;

	// Threshold
	Mat MT,a,b,c;

	// Background Remover object
	FindPeople bg_rem;

	for (int i = 0; i < 1000; ++i) {

		// Get the frame
		video >> frame;

		// Extract people
		Mat fg = bg_rem.find_people(frame);

		namedWindow("Threshold",WINDOW_NORMAL);
		resizeWindow("Threshold", 600, 600);
		imshow("Threshold", fg);

		Mat drawing = bg_rem.find_contours(fg);
		namedWindow("Contours",WINDOW_NORMAL);
		resizeWindow("Contours", 600, 600);
		imshow("Contours", drawing);

		waitKey(1);
	}


	return 0;
}