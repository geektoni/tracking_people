/*
* Written (W) 2018 uriel
*/

#include "Human.h"
#include <cstdlib> // rand()

Human::Human(int id)
{
	this->id = id;

	// Set a random color
	color = Scalar (rand()%255, rand()%255, rand()%255);
}