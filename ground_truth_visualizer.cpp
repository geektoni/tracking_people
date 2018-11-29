/*
* Written (W) 2018 uriel
*/

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>

#include <iostream>
#include <fstream>

#include "Human.h"
#include "utils.h"

using namespace cv;

static int NUM_LINES = 16656;

struct human_frame
{
	human_frame(int id, int fr)
	{
		human = new Human(id);
		frame = fr;
		end = fr;
	}
	Human * human;
	int frame;
	int end;
};

vector<human_frame> humans;

int exists(int id)
{
	for (int i=0; i<humans.size(); i++)
	{
		if (humans[i].human->get_id() == id)
			return i;
	}
	return -1;
}

int main(int argc, char * * argv) {

	// Open the video
	VideoCapture cap("../data/A1_test.mp4");

	// Open the ground truth file
	std::ifstream gt("../data/A1_groundtruthC.txt");

	// check if the file was opened
	if (!gt.is_open()) {
		std::cerr << "File not opened" << std::endl;
		exit(-1);
	}

	// line counter
	int line = 0;

	// frame counter
	int framec = 1;

	int frame_id;
	int user_id;
	float x;
	float y;

	// Add human to the track
	while (line < NUM_LINES) {
		std::string::size_type sz;

		// Get the current line
		std::string current_line;
		getline(gt, current_line);

		// Parse the line and get the tokens
		vector<string> tokens;
		size_t pos = 0;
		while ((pos = current_line.find(",")) != std::string::npos) {
			std::string token = current_line.substr(0, pos);
			tokens.push_back(token);
			current_line.erase(0, pos + 1);
		}
		tokens.push_back(current_line);

		// Convert string to float
		frame_id = stoi(tokens[0]);
		user_id = stoi(tokens[1]);
		x = stof(tokens[2]);
		y = stof(tokens[3]);

		// Create a new human
		human_frame tmp(user_id, frame_id);
		Point2f pt(x, y);

		// Check if it exists
		int id = exists(user_id);
		if (id == -1)
		{
			tmp.human->add_to_trace(pt);
			humans.push_back(tmp);
		} else {
			humans[id].human->add_to_trace(pt);
			humans[id].end = frame_id;
		}

		// Increment the line counter
		line++;
	}

	// Print the traces to the screen
	Mat lines_mask, tracking_lines, frame, tracking;

	while (true)
	{
		// Get the frame
		cap >> frame;

		// Initialize the tracking mask
		lines_mask = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC3);

		frame.copyTo(tracking_lines);
		for (human_frame h : humans) {

			// Check if the user is still there
			if (h.frame <= framec && h.end >= framec)
			{
				if (h.human->get_id() != 10 && h.human->get_id() != 36 && h.human->get_id() != 42)
					continue;

				// Print the line
				int total = h.end-h.frame;
				int i=0, j=1;

				int rc_x = h.human->get_trace()[total-(h.end-framec)-1].x;
				int rc_y = h.human->get_trace()[total-(h.end-framec)-1].y;

				rectangle(lines_mask, Point(rc_x-2, rc_y-2), Point(rc_x+2, rc_y+2), Scalar(0,0,0), -1);
				putText(lines_mask, to_string(h.human->get_id()), Point(rc_x, rc_y), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);

				for (int k=0; k<total-(h.end-framec); k++) {
					cv::line(lines_mask, h.human->get_trace()[i], h.human->get_trace()[j], h.human->get_color(), 3);

					int rc_x = h.human->get_trace()[i].x;
					int rc_y = h.human->get_trace()[i].y;

					j++;
					i++;
				}

				// Print also the current position
				//drawCross(lines_mask, h.human->get_current_position(), h.human->get_color(), 5);
			}
		}

		// Merge the lines and the frame
		tracking = merge_images(tracking_lines, lines_mask);

		// Show the result
		namedWindow("Tracking",WINDOW_NORMAL);
		resizeWindow("Tracking", 600, 600);
		if (!tracking.empty())
			imshow("Tracking", tracking);

		waitKey(20);

		framec++;
	}

	// Delete all humans
	for (auto h : humans)
	{
		delete h.human;
	}


	// Close the ground truth file
	gt.close();

	return 0;
}