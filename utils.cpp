/*
* Written (W) 2018 Giovanni De Toni
*/

#include "utils.h"

cv::Mat merge_images(const cv::Mat & base, const cv::Mat & mask)
{
	cv::Mat base_cp, mask_bw, mask_inv, base_bg, mask_fg, result;

	mask.copyTo(mask_bw);
	base.copyTo(base_cp);

	Mat roi;
	base.copyTo(roi);

	cv::cvtColor(mask_bw, mask_bw, CV_RGB2GRAY);
	cv::threshold(mask_bw, mask_bw, 10, 255, THRESH_BINARY);

	cv::bitwise_not(mask_bw, mask_inv);

	bitwise_and(roi, roi, base_bg, mask_inv);
	bitwise_and(mask, mask, mask_fg, mask_bw);

	add(base_bg, mask_fg, result);

	return result;
}