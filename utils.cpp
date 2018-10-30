/*
* Written (W) 2018 uriel
*/

#include <algorithm>
#include "utils.h"

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