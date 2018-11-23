/*
* Written (W) 2018 Giovanni De Toni
*/

#ifndef TRACKING_PEOPLE_HUMAN_H
#define TRACKING_PEOPLE_HUMAN_H

#include <opencv2/opencv.hpp>

using namespace cv;

class Human {

public:

	/**
	 * Constructor
	 * @param id person unique identifier
	 */
	Human(int id);

	/**
	 * Check if the human placed in that position could be us
	 * @param position the position given
	 * @return true if they are close enough, false otherwise
	 */
	bool is_the_same(const Point2f position, const Mat & contour_histogram=Mat());

	/**
	 * Add the given point to the trace for this user
	 * @param point coordinate point
	 */
	void add_to_trace(const Point2f point);

	/**
	 * Update the position of this human
	 * @param position given position.
	 */
	void update_position(const Point2f position);

	/**
	 * Compute the euclidean distance between the human current position and a point
	 * @param position the new position
	 * @return euclidean distance
	 */
	float get_distance_from(const Point2f position)
	{
		return sqrt(pow(this->current_position.x-position.x, 2)+pow(this->current_position.y-position.y,2));
	}

	/**
	 * Predict step of the Kalman Filter
	 */
	void predict() {
		Mat pred = kalman.predict();
		predicted_point = Point2f(pred.at<float>(0), pred.at<float>(1));
	}

	/**
	 * Error correction step of the Kalman Filter given a measured point
	 * @param measured the measured point
	 * @return the corrected estimate
	 */
	Point2f correct(const Point2f measured)
	{
		Mat_<float> measurement(2,1);
		measurement.setTo(Scalar(0));
		measurement(0) = measured.x;
		measurement(1) = measured.y;
		Mat est = kalman.correct(measurement);
		return Point2f(est.at<float>(0), est.at<float>(1));
	}

	/**
	 * Compute and set the histogram for this human.
	 * @param frame the current frame
	 * @param contour the contour selected
	 * @param _boundRect the bounding rectangle of the contour
	 */
	void set_histogram(const Mat & frame, const std::vector<cv::Point> &contour,
					   const cv::Rect & _boundRect)
	{histogram=Human::compute_histogram(frame, contour,_boundRect);}

	/**
	 * Static method to compute the histogram of a given ROI.
	 * @param frame the current video frame
	 * @param contour the ROI contour
	 * @param _boundRect the bounding box of the contour
	 * @return the histogram
	 */
	static Mat compute_histogram(const Mat & frame, const std::vector<cv::Point> &contour, const cv::Rect & _boundRect);

	/**
 	* Converts a contour to a binary mask.
 	* The parameter mask should be a matrix of type CV_8UC1 with proper
 	* size to hold the mask.
 	* @param contour The contour to convert.
 	* @param mask The Mat where the mask will be written. Must have proper size
 	* and type before callign convertContourToMask.
 	*/
	static void convertContourToMask( const std::vector<cv::Point>& contour, cv::Mat& mask )
	{
		std::vector<std::vector<cv::Point>> contoursVector;
		contoursVector.push_back( contour );
		cv::Scalar white = cv::Scalar(255);
		cv::Scalar black = cv::Scalar(0);
		mask.setTo(black);
		cv::drawContours(mask, contoursVector, -1, white, CV_FILLED);
	}

	/* General getter and setters */
	int get_disappearence() {return this->disappearence;}
	void update_disappearence() {this->disappearence++;}
	void reset_disappearence() {this->disappearence=0;}
	bool is_disappeared() {return this->disappeared;}
	vector<Point2f> get_trace() {return this->trace;}
	Scalar get_color() {return this->color;}
	int get_id() {return this->id;}
	Point2f get_current_position() {return this->current_position;}
	void initialize_kalman(double x, double y);
	KalmanFilter & get_kalman() {return kalman;}
	Point2f get_predicted_point() {return predicted_point;}
	Mat get_histogram() {return histogram;}
	void kill() {this->disappeared = true;}
	bool has_decided() {return this->has_decided_next;}
	void set_decided(bool hasde) {this->has_decided_next=hasde;}

private:

	// The human id
	int id;

	// The human trace
	vector<Point2f> trace;

	// Current position of the human
	Point2f current_position;

	// Histogram of the current person
	Mat_<float> histogram;

	// Acceptable error when computing people position
	double position_error = 40;

	// Color of this human
	Scalar color;

	// Counter to check if the user has exited the scene
	int disappearence;

	// Set to true if the user disappeared
	bool disappeared;

	// Kalman Filter for this user
	KalmanFilter kalman;

	// Prediction point
	Point2f predicted_point;

	// histogram threshold
	double histogram_threshold = 0.2;

	// flag to check if it has get the new point
	bool has_decided_next;

};


#endif //TRACKING_PEOPLE_HUMAN_H
