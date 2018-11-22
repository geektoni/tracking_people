/*
* Written (W) 2018 Giovanni De Toni
*/

#ifndef TRACKING_PEOPLE_SHADOW_REMOVAL_H
#define TRACKING_PEOPLE_SHADOW_REMOVAL_H

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>

#include "Human.h"

class FindPeople
{
public:
	FindPeople(bool preprocess_shadows=true);

	/**
 	* Method that performs background subtraction tasks
 	* in order to find moving elements in the scene.
 	* @param input the input frame
 	* @return a binary matrix with the moving figures highligthed
 	*/
	cv::Mat find_people(const cv::Mat input);

	/**
	 * Find contours of moving objects given a frame
	 * @param input a black and white frame
	 * @return a frame with red contours
	 */
	cv::Mat find_contours(const cv::Mat input, const cv::Mat original_input,
				  bool use_bounding_box,
				  cv::vector<cv::vector<cv::Point>> & _contours,
				  cv::vector<cv::Rect> & _boundRect);

	/**
	 * Track people given their bounding boxes
	 * @param previous previous frame (at time step t-1)
	 * @param current current frame (at time step t)
	 * @param _contours the contours of the people found on the frame
	 * @param _boundRect the bounding boxes computed from the contours
	 * @return the set of next points
	 */
	cv::vector<cv::Point2f> track_people_optical(cv::Mat previous, cv::Mat current,
												 cv::vector<cv::vector<cv::Point>> & _contours,
												 cv::vector<cv::Rect> & _boundRect);

	void track_people_kalman(cv::Mat current, cv::vector<cv::vector<cv::Point>> & _contours,
							 cv::vector<cv::Rect> & _boundRect);

	void update_humans(cv::vector<cv::Point2f> points, int frame_size);

	void update_humans_kalman(cv::vector<cv::Point2f> points, int frame_size);

	cv::vector<Human> return_humans() {return this->humans_tracked;}

	/**
	 * Compute the center of bounding boxes.
	 * @param boundRect a vector containing the bounding boxes
	 * @return a vector with the centers
	 */
	static cv::vector<cv::Point2f> compute_center(const cv::vector<cv::Rect> & boundRect);

	/**
	 * Compute centroids of the contours
	 * @param contours vector of several contours
	 * @return a vector with the computed centroids
	 */
	static cv::vector<cv::Point2f> compute_centroids(const cv::vector<cv::vector<cv::Point>> & contours);

		/**
	 * Converts a contour to a binary mask.
	 * The parameter mask should be a matrix of type CV_8UC1 with proper
	 * size to hold the mask.
	 * @param contour The contour to convert.
	 * @param mask The Mat where the mask will be written. Must have proper size
	 * and type before callign convertContourToMask.
	 */
	void convertContourToMask( const std::vector<cv::Point>& contour, cv::Mat& mask )
	{
		std::vector<std::vector<cv::Point>> contoursVector;
		contoursVector.push_back( contour );
		cv::Scalar white = cv::Scalar(255);
		cv::Scalar black = cv::Scalar(0);
		mask.setTo(black);
		cv::drawContours(mask, contoursVector, -1, white, CV_FILLED);
	}

private:

	/**
 	* This function removes the shadows presents in a
 	* frame of the video. This step is needed to ensure
 	* good detection of the people.
 	* @param frame current video frame
 	*/
	cv::Mat shadow_removal(const cv::Mat frame);

	cv::vector<cv::Rect> generate_bounding_boxes(const cv::vector<cv::vector<cv::Point>> & contours);

	cv::Ptr<cv::BackgroundSubtractor> pGMM;
	int thresh = 128;
	bool preprocess_shadows;

	// Human counter
	int counter;

	// Human found
	cv::vector<Human> humans_tracked;

	// Threshold human disappearence (frames)
	int disappearence_threshold = 200;

	// border threshold
	int border_threshold = 20;
};

#endif //TRACKING_PEOPLE_SHADOW_REMOVAL_H
