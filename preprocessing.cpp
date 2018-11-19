/*
* Written (W) 2018 Giovanni De Toni
*/

#include "preprocessing.h"

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

	// Do a thresholding
	Mat thres = channels[0];
	//medianBlur(thres, thres, 5);
	adaptiveThreshold(thres, thres, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 401, -10);

	return thres;
}

Mat FindPeople::find_people(const Mat input)
{
	// Remove the shadows
	Mat fg = preprocess_shadows ? shadow_removal(input) : input;

	// Apply the GMM
	pGMM->operator()(fg, fg, -1);

	// Remove evetual shadows that were left
	threshold(fg, fg, this->thresh, 255, THRESH_BINARY);

	// Apply opening operator
	morphologyEx(fg, fg, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3,3), Point(-1,-1)), Point(-1,-1), 1);

	return fg;
}

cv::Mat FindPeople::find_contours(const cv::Mat input)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(input, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));

	/// Draw contours
	Mat drawing = Mat::zeros( input.size(), CV_8UC3 );
	for( int i = 0; i< contours.size(); i++ )
	{
		Scalar color = Scalar( 0,0,255);
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
	}
	return drawing;
}
