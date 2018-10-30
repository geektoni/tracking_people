#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "utils.h"

using namespace cv;

int main(int argc, char ** argv) {

	// Parse the command line
	std::vector<string> arguments {"-f"};
	InputParser parser(argc, argv, arguments);
	parser.parse();

	// Open the video and check if it is correct
	VideoCapture video(parser.get("-f"));
	if (!video.isOpened())
		return -1;

	Mat frame;
	for (int i = 0; i < 1000; ++i) {
		video >> frame;
		imshow("Original", frame);
		waitKey(1);
	}


	return 0;
}