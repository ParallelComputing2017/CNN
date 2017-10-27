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

class Logger {

	static bool const debug_level = false;

public:
	static void debug(char* format, ...) {

		if (debug_level) {
			va_list args;
			va_start(args, format);
			vprintf(format, args);
			va_end(args);
		}
	}
};

#endif /* LOGGER_H_ */
