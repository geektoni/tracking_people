/*
* Written (W) 2018 uriel
*/

#include "Human.h"
#include <cstdlib> // rand()
#include <iostream>

Human::Human(int id)
{
	this->id = id;
	this->disappeared = false;
	this->disappearence = 0;

	// Set a random color
	color = Scalar (rand()%255, rand()%255, rand()%255);
}

bool Human::is_the_same(const Point2f position)
{
	// Compute distance between points
	float distance = sqrt(pow(this->current_position.x-position.x, 2)+pow(this->current_position.y-position.y,2));

	std::cout << distance << std::endl;

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