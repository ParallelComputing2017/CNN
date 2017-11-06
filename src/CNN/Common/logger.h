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
#include <string.h>

using namespace std;

class Logger {

private:
	static bool const debug_level = true;
	static bool const info_level = true;

	static void write(string format, ...) {
		va_list args;
		va_start(args, format);
		vprintf(format.c_str(), args);
		va_end(args);
	}

public:
	static void debug(string format, ...) {

		if (debug_level) {
			va_list args;
			va_start(args, format);
			write("DEBUG \t" + format, args);
			va_end(args);
		}
	}

	static void info(string format, ...) {

		if (info_level) {
			va_list args;
			va_start(args, format);
			write("INFO \t" + format, args);
			va_end(args);
		}
	}
};

#endif /* LOGGER_H_ */