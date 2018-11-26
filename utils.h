/*
* Written (W) 2018 Giovanni De Toni
*/

#ifndef TRACKING_PEOPLE_UTILS_H
#define TRACKING_PEOPLE_UTILS_H

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>

using namespace std;
using namespace cv;

/**
 * Merge to images together (mask on top of base)
 * @param base the original image
 * @param mask the mask we want to glue on top of the original image
 * @return the merged image
 */
cv::Mat merge_images(const cv::Mat & base, const cv::Mat & mask);


#endif //TRACKING_PEOPLE_UTILS_H
