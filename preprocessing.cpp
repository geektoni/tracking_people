/*
* Written (W) 2018 uriel
*/

#include "preprocessing.h"

using namespace cv;


void erode_shape(Mat &frame, Mat &output, int shape)
{
	Mat str = getStructuringElement(shape, Size(3,3), Point(0,0));
	erode(frame, output, str);
}

/**
 * First dilate, then erode
 */
void closing(Mat & frame)
{
	Mat dil;

	Mat str = getStructuringElement(MORPH_ELLIPSE, Size(4,4), Point(0,0));

	dilate(frame, dil, str);
	erode(dil, frame, str);
}


void opening(Mat &frame, Mat &output, int shape)
{
	Mat dil;

	Mat str = getStructuringElement(shape, Size(3,3), Point(0,0));

	erode(frame, dil, str);
	dilate(dil,output, str);
}

/**
 * https://lowweilin.wordpress.com/2014/08/07/image-background-and-shadow-removal/
 */
void FindPeople::shadow_removal(Mat & frame)
{
	Mat tmp_frame;

	// Convert the frame from RGB to the HSV color space
	cvtColor(frame, tmp_frame, CV_RGB2HSV);

	// Loop over the image pixels and set the V to a fixed color
	for(int y=0;y<tmp_frame.rows;y++)
	{
		for(int x=0;x<tmp_frame.cols;x++)
		{
			// get pixel
			Vec3b color = tmp_frame.at<Vec3b>(Point(x,y));

			// Fix the V component
			color[2] = 127;

			// set pixel
			tmp_frame.at<Vec3b>(Point(x,y)) = color;
		}
	}

	//Now convert back to RGB and we are done
	cvtColor(tmp_frame, frame, CV_HSV2RGB);

}

FindPeople::FindPeople() {
	this->pGMM = new BackgroundSubtractorMOG2(300, 600, true);
}

Mat FindPeople::find_people(const Mat input)
{
	Mat fg;

	/// Convert to grayscale
	//cvtColor(input,fg, CV_BGR2GRAY);

	/// Apply Histogram Equalization
	//equalizeHist(fg,fg);

	pGMM->operator()(input, fg, -1);

	// Partly remove the shadows
	threshold(fg, fg, this->thresh, 255, THRESH_BINARY);

	//GaussianBlur(fg, fg, Size(3, 3), 0, 0);

	// Apply opening and closing operators operator
	//morphologyEx(fg, fg, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(1,1), Point(-1,-1)), Point(-1,-1), 1);
	morphologyEx(fg, fg, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(6,6), Point(-1,-1)), Point(-1,-1), 1);

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
