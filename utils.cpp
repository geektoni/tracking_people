/*
* Written (W) 2018 uriel
*/

#include <algorithm>
#include "utils.h"
#include <iostream>

InputParser::InputParser(int argc, char ** argv, vector<string> parameters)
{
	this->argc = argc;
	this->argv = argv;
	this->parameters = parameters;
}

void InputParser::parse() {
	for (int i=1; i<argc; i++)
	{
		string tmp_string(argv[i]);
		auto it=find(this->parameters.begin(), this->parameters.end(), tmp_string);
		if (it != this->parameters.end())
		{
			string tmp_arg (argv[i+1]);
			this->parsed_argv.insert(make_pair(tmp_string, tmp_arg));
			i++;
		}
	}
}

string InputParser::get(string key) {
	auto it = this->parsed_argv.find(key);

	if (it == this->parsed_argv.end())
		return "";

	return it->second;
}

cv::Mat merge_images(const cv::Mat & base, const cv::Mat & mask)
{
	cv::Mat base_cp, mask_bw, mask_inv, base_bg, mask_fg, result;

	mask.copyTo(mask_bw);
	base.copyTo(base_cp);

	Mat roi = Mat::zeros(base.size(), CV_8UC3);

	cv::cvtColor(mask_bw, mask_bw, CV_RGB2GRAY);
	cv::threshold(mask_bw, mask_bw, 10, 255, THRESH_BINARY);

	cv::bitwise_not(mask_bw, mask_inv);

	bitwise_and(roi, roi, base_bg, mask_inv);
	bitwise_and(base, base_cp, mask_fg, mask_bw);

	add(base_bg, mask_fg, result);

	return result;
}