/*
* Written (W) 2018 uriel
*/

#include "Human.h"
#include <cstdlib> // rand()
#include <iostream>
#include <opencv/cv.h>

Human::Human(int id)
{
	this->id = id;
	this->disappeared = false;
	this->disappearence = 0;

	// Set a random color
	color = Scalar (rand()%255, rand()%255, rand()%255);

	// Initialize the Kalman (we use also the accelleration)
	kalman = KalmanFilter(6,2,0);
}

bool Human::is_the_same(const Point2f position)
{
	// Compute distance between points
	float distance = sqrt(pow(this->current_position.x-position.x, 2)+pow(this->current_position.y-position.y,2));

	return !(distance > this->position_error);
}

void Human::add_to_trace(const Point2f point)
{
	this->trace.push_back(point);
}

void Human::update_position(const Point2f position)
{
	this->current_position = position;
}

void Human::initialize_kalman(double x, double y)
{
	// Set up the initial state of the Kalman filter
	// (we do not set the velocity nor the acceleration).
	kalman.statePre.at<float>(0) = x;
	kalman.statePre.at<float>(1) = y;
	kalman.statePre.at<float>(2) = 0;
	kalman.statePre.at<float>(3) = 0;
	kalman.statePre.at<float>(4) = 0;
	kalman.statePre.at<float>(5) = 0;

	// We need then to define the transition matrix.
	// It has to be 4x4 in order to get the following equations:
	// -> x = x_0 + t*v_x + 1/2*a_x*t^2
	// -> y = y_0 + t*v_y + 1/2*a_y*t^2
	// -> v_xt+1 = v_tx+a_xt
	// -> v_ut+1 = v_ty+a_yt
	// -> a_xt+1 = a_xt
	// -> a_yt+1 = a_yt+1
	// Therefore, T must be:
	// 1 0 1 0 1/2 0
	// 0 1 0 1 0 1/2
	// 0 0 1 0 0 0
	// 0 0 0 1 0 0
	// 0 0 0 0 1 0
	// 0 0 0 0 0 1
	// such that T*S will give us those equations.
	kalman.transitionMatrix = *(Mat_<float>(6,6) <<
												 1,0,1,0,0.5,0,
			0,1,0,1,0,0.5,
			0,0,1,0,1,0,
			0,0,0,1,0,1,
			0,0,0,0,1,0,
			0,0,0,0,0,1);

	// Set the measurement function to the identity
	// and to the process noise covariance. The covariance
	// will have slightly smaller values on the diagonal.
	setIdentity(kalman.measurementMatrix);
	setIdentity(kalman.processNoiseCov, Scalar::all(1e-5));
	setIdentity(kalman.measurementNoiseCov, Scalar::all(1e-3));
	setIdentity(kalman.errorCovPost, Scalar::all(0.1));
}

Mat Human::compute_histogram(const Mat & frame, const std::vector<cv::Point> &contour) {

	// Compute the mask for the given contour
	Mat mask;
	Human::convertContourToMask(contour, mask);

	int channels[] = {1,2,3};

	// Quantize the hue to 30 levels
	// and the saturation to 32 levels
	int histSize[] = {100, 100, 100};

	// Color ranges
	float range[] = {0,256};
	const float* ranges[] = {range, range, range};

	// Compute the actual histogram
	Mat histo;
	calcHist( &frame, 1, channels, mask,
			  histo, 2, histSize, ranges,
			  true, // the histogram is uniform
			  false );
	return histo;
}