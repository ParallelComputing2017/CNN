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

int main(int argc, char *argv[]) {

	bool fullTestRun = true;
	int threads = 3;
	string self(argv[0]);
	string mode = "cuda";

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

	ParallelCNN cnn;
	vector<layer_t*> layers;

	if (mode.compare("posix") == 0 || all) {
		timer.start();
		layers = cnn.posix(threads);
		timer.stop();
	}
	if (mode.compare("openmp") == 0 || all) {
		timer.start();
		layers = cnn.openMP(threads);
		timer.stop();
	}
	if (mode.compare("cuda") == 0 || all) {
		timer.start();
		layers = cnn.cuda(threads);
		timer.stop();
		fullTestRun = false;
	}
	if (mode.compare("sequential") == 0) {
		timer.start();
		layers = cnn.sequential();
		timer.stop();
	}

	Logger::info("Training time: %s", timer.format(3, "%ws").c_str());

	// Testing
	if (fullTestRun) {
		timer.start();
		fullTest(layers);
		timer.stop();
		Logger::info("Full test time: %s", timer.format(3, "%ws").c_str());
	}

	timer.start();
	int digit = singleTest(layers);
	Logger::info("Single test. Digit: %d,  time: %s", digit,
			timer.format(3, "%ws").c_str());
	timer.stop();

	return 0;

}

