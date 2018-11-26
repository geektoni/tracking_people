#include <iostream>
#include <fstream>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>

#include "utils.h"
#include "preprocessing.h"

using namespace cv;
using namespace std;

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
	Mat frame, previous, lines_mask, tracking_lines, tracking;

	// Foreground mask (generated by GMM)
	Mat fg;

	// result
	Mat fg_copy;

	// Background Remover object
	FindPeople bg_rem;

	// Contours;
	vector<vector<Point>> contours;
	vector<Rect> boundRect;

	// Open output file
	ofstream output("./people_track.csv");

	// Print the header
	output << "frame,id,X,Y" << endl;

	// Frame counter
	int frame_counter=1;

	while (true) {

		// Get the frame
		video >> frame;

		// Initialize the tracking mask
		lines_mask = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC3);

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

		// Start tracking only if we reached the starting frame
		//if (frame_counter<1)
		//{
		//	frame_counter++;
		//	continue;
		//}

		if (!previous.empty())
		{
			//bg_rem.track_people_optical(previous, frame, contours, boundRect);
			bg_rem.track_people_kalman(frame, contours, boundRect, frame_counter);

			// Get the tracked humans
			auto humans = bg_rem.return_humans();

			// Print a line between the points and the result
			for (Human h : humans) {

				// Check if the user is still there
				if (!h.is_disappeared())
				{
					// Print the user id on top of all the humans detected
					int rc_x = h.get_trace()[h.get_trace().size()-1].x;
					int rc_y = h.get_trace()[h.get_trace().size()-1].y;
					putText(lines_mask, to_string(h.get_id()), Point(rc_x, rc_y), FONT_HERSHEY_SIMPLEX,
							1, Scalar(255,255,255), 2);

					// Print the human track
					for (int i=0, j=1; j<h.get_trace().size();)
					{
						line(lines_mask, h.get_trace()[i], h.get_trace()[j], h.get_color(), 3);
						j++;
						i++;
					}

					// Print also a cross indicating the current positions
					drawMarker(lines_mask, h.get_current_position(), h.get_color());
				}

			}

			// Merge the lines and the frame
			tracking = merge_images(tracking_lines, lines_mask);

			// Update the previous frame
			frame.copyTo(previous);
		}

		// Update for the first thame the previous frame
		if (previous.empty())
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
		if (!tracking.empty())
			imshow("Tracking", tracking);

		waitKey(1);

		// Print the current position for each human
		for (Human h : bg_rem.return_humans())
		{
			if (!h.is_disappeared())
			{
				// Print the value for this frame
				output << frame_counter
					   << "," << h.get_id()
					   << "," << h.get_current_position().x
					   << "," << h.get_current_position().y << endl;
			}
		}

		// Increment the frame counter
		frame_counter++;

	}

	// Close the file
	output.close();

	return 0;
}