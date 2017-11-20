/*
 * logger.h
 *
 *  Created on: 27/10/2017
 *      Author: sebas
 */

#ifndef LOGGER_H_
#define LOGGER_H_

#include <stdio.h>
#include <stdarg.h>
#include <string>
#include <thread>
#include <sstream>

using namespace std;

class Logger {

private:
	static bool const debug_level = false;
	static bool const info_level = true;

	static string getThreadId() {
		std::thread::id this_id = std::this_thread::get_id();

		thread::id myid = this_thread::get_id();
		hash<thread::id> hasher;
		stringstream ss;
		ss << hasher(myid);
		string tId = ss.str();
		return tId.substr(0, 5);
	}

public:
	static void debug(string format, ...) {

		string tId = getThreadId();

		if (debug_level) {
			va_list args;
			va_start(args, format);
			vprintf((tId + "  DEBUG  " + format + "\n").c_str(), args);
			va_end(args);
		}
	}

	static void info(string format, ...) {

		string tId = getThreadId();

		if (info_level) {
			va_list args;
			va_start(args, format);
			vprintf((tId + "  INFO  " + format + "\n").c_str(), args);
			va_end(args);
		}
	}
};

#endif /* LOGGER_H_ */
