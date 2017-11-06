/*
 * main.cpp
 *
 *  Created on: 10/08/2017
 *      Author: juan
 */
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>

#include <boost/timer/timer.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace boost::timer;

#include "parallel_cnn.h"

void printLog(string mode, cpu_timer timer, int result);

int main(int argc, char *argv[]) {

	int threads = 3;
	string self(argv[0]);
	string mode = "openmp";

	if (argc != 3) { // argc should be 3 for correct execution
		printf("Usage: <mode> <num_threads>\n");
		printf("\tModes: all, posix, openmp, sequential, cuda \n");
	} else {
		mode = argv[1];
		threads = atoi(argv[2]);
	}

	cpu_timer timer;

	printf("Running mode: %s-%i \n", mode.c_str(), threads);

	bool all = (mode.compare("all") == 0);

	if (mode.compare("posix") == 0 || all) {

		timer.start();
		int digit = posix(threads);
		timer.stop();

		printLog("Posix", timer, digit);
	}
	if (mode.compare("openmp") == 0 || all) {

		timer.start();
		int digit = openMP(threads);
		timer.stop();

		printLog("OpenMP", timer, digit);

	}
	if (mode.compare("cuda") == 0 || all) {

		timer.start();
		int digit = cuda(threads);
		timer.stop();

		printLog("CUDA", timer, digit);

	}
	if (mode.compare("sequential") == 0) {

		timer.start();
		int digit = sequential();
		timer.stop();

		printLog("Single", timer, digit);
	}

	return 0;

}

void printLog(string mode, cpu_timer timer, int result) {

	printf("%s\t time: %s\t result: %i \n", mode.c_str(),
			timer.format(3, "%ws").c_str(), result);
}

