/*
* Written (W) 2018 uriel
*/

#ifndef TRACKING_PEOPLE_HUMAN_H
#define TRACKING_PEOPLE_HUMAN_H

#include <opencv/cxcore.h>

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
	bool is_the_same(const Point2f position);

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

private:

	// The human id
	int id;

	// The human trace
	vector<Point2f> trace;

	// Current position of the human
	Point2f current_position;

	// Histogram of the current person (may be useless though)
	Mat histogram;

	// Acceptable error when computing people position
	double position_error = 2.0;

	// Color of this human
	Scalar color;

};


#endif //TRACKING_PEOPLE_HUMAN_H
