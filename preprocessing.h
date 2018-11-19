/*
* Written (W) 2018 Giovanni De Toni
*/

#ifndef TRACKING_PEOPLE_SHADOW_REMOVAL_H
#define TRACKING_PEOPLE_SHADOW_REMOVAL_H

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>

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
	cv::Mat find_contours(const cv::Mat input);

private:

	/**
 	* This function removes the shadows presents in a
 	* frame of the video. This step is needed to ensure
 	* good detection of the people.
 	* @param frame current video frame
 	*/
	cv::Mat shadow_removal(const cv::Mat frame);


	cv::Ptr<cv::BackgroundSubtractor> pGMM;
	int thresh = 128;
	bool preprocess_shadows;
};

#endif //TRACKING_PEOPLE_SHADOW_REMOVAL_H
