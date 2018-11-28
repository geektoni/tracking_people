/*
* Written (W) 2018 Giovanni De Toni
*/

#ifndef TRACKING_PEOPLE_SHADOW_REMOVAL_H
#define TRACKING_PEOPLE_SHADOW_REMOVAL_H

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>

#include "Human.h"

/**
 * This class implements a people detector procedure.
 */
class FindPeople
{
public:

	/**
	 * Constructor.
	 * @param preprocess_shadows set to true if we want to do shadow removal
	 * before doing the background subtraction.
	 */
	FindPeople(bool preprocess_shadows=true);

	/**
 	* Method that performs background subtraction tasks
 	* in order to find moving elements in the scene.
 	* @param input the input frame.
 	* @return a binary matrix with the moving figures highligthed.
 	*/
	cv::Mat find_people(const cv::Mat input);

	/**
	 * Given a binary matrix, this methods compute the contours of each
	 * of the white blob present in the input matrix. It will also compute
	 * the bounding boxes for each of these blobs. Moreover, it will generate
	 * a new matrix on which each of the contours will be printed. The returned
	 * matrix will have also a counter displaying how many figure we have found
	 * in the current frame.
	 * @param input the binary matrix.
	 * @param original_input the original input from which the binary matrix was computed.
	 * @param use_bounding_box flag to set if we want to use the bounding boxes.
	 * @param _contours this vector will contain the found contours.
	 * @param _boundRect this vector will contain the found bounding boxes.
	 * @param people_count this integer will contain the people count for each frame.
	 * @return the original_input plus the contours+bounding boxes+people counter.
	 */
	cv::Mat find_contours(const cv::Mat input, const cv::Mat original_input,
				  bool use_bounding_box,
				  cv::vector<cv::vector<cv::Point>> & _contours,
				  cv::vector<cv::Rect> & _boundRect, int & people_count);

	/**
	 * Track people given their bounding boxes (using Lucas-Kanade Optical Flow).
	 * @param previous previous frame (at time step t-1)
	 * @param current current frame (at time step t)
	 * @param _contours the contours of the people found on the frame
	 * @param _boundRect the bounding boxes computed from the contours
	 */
	void track_people_optical(cv::Mat previous, cv::Mat current,
												 cv::vector<cv::vector<cv::Point>> & _contours,
												 cv::vector<cv::Rect> & _boundRect);

	/**
	 * Track people given their contours and by using the Kalman Filter.
	 * @param current the current frame.
	 * @param _contours the contours found in the current frame.
	 * @param _boundRect the bounding boxes of the contours found.
	 * @param frame_count the current frame number.
	 */
	void track_people_kalman(cv::Mat current, cv::vector<cv::vector<cv::Point>> & _contours,
							 cv::vector<cv::Rect> & _boundRect, const int frame_count);

	/**
	 * Track people given their contours and by using their centroid and histogram.
	 * @param current the current frame.
	 * @param _contours the contours found in the current frame.
	 * @param _boundRect the bounding boxes of the contours found.
	 * @param frame_count the current frame number.
	 */
	void track_people_simple(cv::Mat current, cv::vector<cv::vector<cv::Point>> &_contours,
							 cv::vector<cv::Rect> &_boundRect, const int frame_count);

	/**
	 * Return the humans tracked up to now.
	 * @return a vector containing the humans.
	 */
	cv::vector<Human> return_humans() {return this->humans_tracked;}

	/**
	 * Compute the center of bounding boxes.
	 * @param boundRect a vector containing the bounding boxes
	 * @return a vector with the centers
	 */
	static cv::vector<cv::Point2f> compute_center(const cv::vector<cv::Rect> & boundRect);

	/**
	 * Compute centroids of the contours.
	 * @param contours vector of several contours.
	 * @return a vector with the computed centroids.
	 */
	static cv::vector<cv::Point2f> compute_centroids(const cv::vector<cv::vector<cv::Point>> & contours);

private:

	/**
 	* This function removes the shadows presents in a
 	* frame of the video. This step is needed to ensure
 	* good detection of the people.
 	* @param frame current video frame
 	*/
	cv::Mat shadow_removal(const cv::Mat frame);

	/**
	 * Given a set of contours, it will return the bouding boxes for each of these contours.
	 * @param contours a vector of contours.
	 * @return a vector of bounding boxes.
	 */
	cv::vector<cv::Rect> generate_bounding_boxes(const cv::vector<cv::vector<cv::Point>> & contours);

	/**
	 * Update human position.
	 * @param points the centers of the blob found.
	 * @param frame_size the size of the video (number of columns of the frame).
	 */
	void update_humans(cv::vector<cv::Point2f> points, int frame_size);

	/**
	 * Update human position using the Kalman filter.
	 * @param current the current frame.
	 * @param points the centers of the blob found.
	 * @param frame_size the size of the video (number of columns of the frame).
	 * @param _contours the contours found in the current frame.
	 * @param _boundRect the bounding boxes of the contours found.
	 * @param frame_count the current frame number.
	 */
	void update_humans_kalman(cv::Mat current, cv::vector<cv::Point2f> points, int frame_size,
							  cv::vector<cv::vector<cv::Point>> &_contours,
							  const cv::vector<cv::Rect> & _boundRect, const int frame_count);

	/**
	 * Update human position using a simple region-based approach.
	 * @param current the current frame.
	 * @param points the centers of the blob found.
	 * @param frame_size the size of the video (number of columns of the frame).
	 * @param _contours the contours found in the current frame.
	 * @param _boundRect the bounding boxes of the contours found.
	 * @param frame_count the current frame number.
	 */
	void update_humans_simple(cv::Mat current,
							  cv::vector<cv::Point2f> points, int frame_size,
							  cv::vector<cv::vector<cv::Point>> &_contours,
							  const cv::vector<cv::Rect> & _boundRect, const int frame_count);

	/**
	 * The background subtraction instance.
	 */
	cv::Ptr<cv::BackgroundSubtractor> pGMM;

	/**
	 * The shadow removal color threshold.
	 */
	int thresh = 128;

	/**
	 * Flag to check if we want to remove the shadows.
	 */
	bool preprocess_shadows;

	// Human counter
	int counter;

	// Humans found
	cv::vector<Human> humans_tracked;

	// Threshold human disappearence (frames counter)
	int disappearence_threshold = 200;

	// border threshold
	int border_threshold = 20;

	// initial frame detection threshold
	int frame_detection_threshold = 10;
};

#endif //TRACKING_PEOPLE_SHADOW_REMOVAL_H
