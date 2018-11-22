/*
* Written (W) 2018 uriel
*/

#ifndef TRACKING_PEOPLE_UTILS_H
#define TRACKING_PEOPLE_UTILS_H

#include <vector>
#include <string>
#include <map>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>

using namespace std;
using namespace cv;

class InputParser
{
public:
	InputParser(int argc, char ** argv, vector<string> parameters);

	/**
	 * Parse the input given by the program. It will
	 * populate the parsed_argv map.
	 */
	void parse();

	/**
	 * Get the value of the map entry defined by key.
	 * @param key key of entry
	 * @return the value corresponding to the key, or an empty string
	 */
	string get(string key);

private:
	int argc; // Number of arguments
	char ** argv; // Array of strings
	vector<string> parameters; // Parameters we want to find
	map<string, string> parsed_argv; // Parsed input string
};

cv::Mat merge_images(const cv::Mat & base, const cv::Mat & mask);


#endif //TRACKING_PEOPLE_UTILS_H
