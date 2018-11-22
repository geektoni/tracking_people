/*
* Written (W) 2018 Giovanni De Toni
*/

#include "preprocessing.h"
#include <string>
#include <iostream>

using namespace cv;

FindPeople::FindPeople(bool preprocess_shadows) {
	this->pGMM = new BackgroundSubtractorMOG2(300, 350, true);
	this->preprocess_shadows = preprocess_shadows;
}

/**
 * https://lowweilin.wordpress.com/2014/08/07/image-background-and-shadow-removal/
 * https://stackoverflow.com/questions/20542352/automatic-approach-for-removing-colord-object-shadow-on-white-background
 */
Mat FindPeople::shadow_removal(const Mat frame)
{
	Mat tmp_frame;

	// Convert the frame from RGB to the HSV color space
	cvtColor(frame, tmp_frame, CV_RGB2HSV);

	// Extract the channels
	vector<Mat> channels;
	split(tmp_frame, channels);

	// Return only the H channel (remove shadows);
	return channels[0];
}

Mat FindPeople::find_people(const Mat input)
{
	// Remove the shadows
	Mat fg = preprocess_shadows ? shadow_removal(input) : input;

	// Apply the GMM
	pGMM->operator()(fg, fg, -1);

	// Remove evetual shadows that were left
	threshold(fg, fg, this->thresh, 255, THRESH_BINARY);

	// Apply opening and dilation operator
	morphologyEx(fg, fg, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3,3), Point(-1,-1)), Point(-1,-1), 1);
	morphologyEx(fg, fg, MORPH_DILATE, getStructuringElement(MORPH_RECT, Size(3,3), Point(-1,-1)), Point(-1,-1), 2);

	return fg;
}

// https://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/bounding_rects_circles/bounding_rects_circles.html
vector<Rect> FindPeople::generate_bounding_boxes(const vector<vector<Point>> & contours)
{
	vector<vector<Point>> contours_poly( contours.size() );
	vector<Rect> boundRect( contours.size() );
	vector<Point2f> center( contours.size() );

	for( int i = 0; i < contours.size(); i++ )
	{
		approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
		boundRect[i] = boundingRect( Mat(contours_poly[i]) );
	}

	return boundRect;
}

cv::Mat FindPeople::find_contours(const cv::Mat input, const cv::Mat original_input,
								  bool use_bounding_box,
								  vector<vector<Point>> & _contours,
								  vector<Rect> & _boundRect) {

	vector<vector<Point>> contours, filtered_contours;
	vector<Vec4i> hierarchy;

	// Find people contours
	findContours(input, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	// Filter contours in order to avoid false positives
	for (vector<Point> c : contours)
	{
		// Skip the contour if it is too small
		if (contourArea(c) < 400) {
			continue;
		}
		filtered_contours.push_back(c);
	}

	// If we want bounding boxes instead of just the contours
	vector<Rect> boundRect(filtered_contours.size());
	if (use_bounding_box) {
		boundRect = this->generate_bounding_boxes(filtered_contours);
	}

	// Draw contours and count possible "peoples"
	int total_people_count = 0;
	Mat drawing; original_input.copyTo(drawing);

	for (int i = 0; i < filtered_contours.size(); i++) {

		Scalar color_bbox = Scalar(0, 0, 255);
		Scalar color_cont = Scalar(255,255,255);

		// If we want the bounding boxes then we will use them
		if (use_bounding_box) {
			rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color_bbox, 2, 8, 0);
		}

		// Draw the actual contours
		drawContours(drawing, filtered_contours, i, color_cont, 2, 8, hierarchy, 0, Point());

		// Increase the people counter;
		total_people_count++;
	}

	// Add counter showing how many people are in the image
	std::string total_people = "People Count: " + std::to_string(total_people_count);
	rectangle(drawing, Point(0, 0), Point(300, 70), Scalar(0,0,0), -1);
	putText(drawing, total_people, Point(10, 45), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);

	// Copy the contours and the rectangle
	// The rectangles' centres can be used as features points
	// for the optical flow algorithm. Contours are provided if
	// we need other informations
	_contours = filtered_contours;
	_boundRect = boundRect;

	return drawing;
}

vector<Point2f> FindPeople::track_people_optical(cv::Mat previous, cv::Mat current, cv::vector<cv::vector<cv::Point>> & _contours,
							 cv::vector<cv::Rect> & _boundRect) {

	vector<uchar> status;
	vector<float> err;

	// Comput centroids for each of the points
	// and store them into an array.
	vector<Point2f> points = compute_center(_boundRect);

	// This array will contain the next points
	vector<Point2f> result(points.size());

	// Compute the flow (skip if there are no points to track)
	if (points.size() != 0)
		calcOpticalFlowPyrLK(previous, current, points, result, status, err);

	return result;
}

vector<Point2f> FindPeople::compute_center(const cv::vector<cv::Rect> & _boundRect)
{
	vector<Point2f> points;
	for (int i = 0; i < _boundRect.size(); i++)
	{
		float cx = _boundRect[i].x+_boundRect[i].width/2;
		float cy = _boundRect[i].y+_boundRect[i].height/2;
		points.push_back(Point2f(cx, cy));
	}
	return points;
}