/*
* Written (W) 2018 Giovanni De Toni
*/

#include <iostream>
#include <fstream>
#include <memory>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>

#include "utils.h"
#include "FindPeople.h"

using namespace cv;
using namespace std;

const char * keys = "{h | help | | Print this message.}"
		"{f |file| . | Path to the video that has to be analyzed.}"
		"{a |alg| kalman | Which tracking algorithm will be used, can be 'opticalflow', 'kalman', 'simple'. Default is 'kalman'.}"
		"{s|start | 1 | Start to track/detect objects only from a specific frame. Default is 1.}"
		"{sh|remove_shadow | true | Choose if we want to remove shadow with HSV or just use MOG2 capabilities.}"
		"{save | save_frame | -1 | Save the frame specified by the number to disk.}"
		"{u | user | -1 | Save tracking data only for this specific user.}";

int main(int argc, char ** argv) {

	// Intialize the command line parser
	CommandLineParser parser(argc, argv, keys);

	// Get the path to the video file
	string video_path = parser.get<string>("f");

	// Get the starting frame
	int starting_frame = parser.get<int>("start");

	// Get which algorithm we want to use
	string track_algo = parser.get<string>("alg");

	// Remove the shadows using HSV
	bool remove_shadow = parser.get<bool>("remove_shadow");

	// Which frame need to be saved
	int save_frame = parser.get<int>("save_frame");

	int selected_one = parser.get<int>("user");

	// Open the video and check if it is correct
	// otherwise return with an error.
	VideoCapture video(video_path);
	if (!video.isOpened()) {
		delete keys;
		return -1;
	}

	// Save the video on an output file
	VideoWriter output_video("output.avi",CV_FOURCC('M','J','P','G'),10, Size(1280, 720));

	// Current frame
	Mat frame, previous, lines_mask, tracking_lines, tracking;

	// Foreground mask (generated by GMM)
	Mat fg;

	// result
	Mat fg_copy;

	// Background Remover object
	FindPeople bg_rem(remove_shadow);

	// Contours;
	vector<vector<Point>> contours;
	vector<Rect> boundRect;

	// Open output file
	ofstream output("./people_track.csv");

	// Open people count file
	ofstream output_count("./people_count.csv");

	// Print the header
	output << "frame,id,X,Y" << endl;
	output_count << "frame,count" << endl;

	// Frame counter
	int frame_counter=1;

	// People counter
	int people_counter=0;

	while (true) {

		// Get the frame
		video >> frame;

		// Start tracking/detecting only if we reached the starting frame
		if (frame_counter<starting_frame)
		{
			frame_counter++;
			continue;
		}

		// Initialize the tracking mask
		lines_mask = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC3);

		// If we reach the end of the video, we exit
		if (frame.empty())
			exit(0);

		// Extract people and copy the frame (prevent further modifications)
		Mat fg = bg_rem.find_people(frame);
		fg.copyTo(fg_copy);

		// Find contours and boundin boxes
		Mat drawing = bg_rem.find_contours(fg, frame, true, contours, boundRect, people_counter);

		// Save to file the people count
		output_count << frame_counter << "," << people_counter << endl;

		// Get the bounding rectangle centers
		vector<Point2f> current_centers = FindPeople::compute_center(boundRect);
		vector<Point2f> next_centers;

		// The tracking is performed starting from the
		// second frame. We also do the tracking by measuring the
		// displacement each 5 frame to have less noise.
		frame.copyTo(tracking_lines);

		// Select which algorithm to use for tracking
		if (track_algo.compare("kalman")==0)
		{
			bg_rem.track_people_kalman(frame, contours, boundRect, frame_counter);
		} else if (track_algo.compare("simple")==0){
			bg_rem.track_people_simple(frame, contours, boundRect, frame_counter);
		} else {
			if (!previous.empty())
			{
				bg_rem.track_people_optical(previous, frame, contours, boundRect, frame_counter);
			}
		}

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
						1, Scalar(0,0,255), 2);

				// Print the human track
				for (int i=0, j=1; j<h.get_trace().size();)
				{
					line(lines_mask, h.get_trace()[i], h.get_trace()[j], h.get_color(), 3);
					j++;
					i++;
				}

				// Print also a cross indicating the current positions
				//drawMarker(lines_mask, h.get_current_position(), h.get_color());
			}

		}

		// Merge the lines and the frame
		tracking = merge_images(drawing, lines_mask);

		// Update the previous frame
		frame.copyTo(previous);

		// Save video to disk
		output_video.write(tracking);

		// Print everything on screen
		//namedWindow("Threshold",WINDOW_NORMAL);
		//resizeWindow("Threshold", 600, 600);
		//imshow("Threshold", fg_copy);

		//namedWindow("Detect",WINDOW_NORMAL);
		//resizeWindow("Detect", 600, 600);
		//imshow("Detect", drawing);

		namedWindow("Result",WINDOW_NORMAL);
		resizeWindow("Result", 600, 600);
		if (!tracking.empty())
			imshow("Result", tracking);

		// Save to disk the specified frame
		if (save_frame == frame_counter)
		{
			imwrite("./shadows.png", fg_copy);
			imwrite("./detection.png", drawing);
			if(!tracking.empty())
				imwrite("./tracking.png", tracking);
		}


		waitKey(1);

		// Print the current position for each human
		for (Human h : bg_rem.return_humans())
		{
			if (!h.is_disappeared())
			{
				if (h.get_id() == selected_one && selected_one != -1) {
					// Print the value for this frame
					output << frame_counter
						   << "," << h.get_id()
						   << "," << h.get_current_position().x
						   << "," << h.get_current_position().y << endl;
				}
			}
		}

		// Increment the frame counter
		frame_counter++;

	}

	// Close the file
	output.close();

	// Free the pointer
	delete keys;

	return 0;
}