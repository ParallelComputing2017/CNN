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

#include "NeuralNetwork/NeuralNetwork.hpp"

#include "CNN/MNIST.h"

void writeCSV(string program, int threads, float runningTime);
void printLog(string mode, cpu_timer timer, int result);

int main(int argc, char *argv[]) {

	int threads = 3;
	string self(argv[0]);
	string mode = "openmp";

	if (argc != 3) { // argc should be 3 for correct execution
		printf("Usage: Posix <mode_name> <num_threads>\n");
		printf("\tModes: all, posix, openmp, single \n");
	} else {
		mode = argv[1];
		threads = atoi(argv[2]);
	}

	cpu_timer timer;

	printf("Using %i threads \n", threads);

	bool all = (mode.compare("all") == 0);

	if (mode.compare("posix") == 0 || all) {

		timer.start();
		timer.stop();

		printLog("Posix", timer, 0);
	}
	if (mode.compare("openmp") == 0 || all) {

		timer.start();
		int digit = openMP(threads);
		timer.stop();

		printLog("OpenMP", timer, digit);

	}
	if (mode.compare("single") == 0) {

		timer.start();
		int digit = mainExample();
		timer.stop();

		printLog("Single", timer, digit);
	}

	return 0;

}

void printLog(string mode, cpu_timer timer, int result) {

	printf("%s\t time: %s\t result: %i \n", mode.c_str(),
			timer.format(3, "%ws").c_str(), result);
}

void writeCSV(string program, int threads, float seconds_runningTime) {

	ofstream myfile;

	myfile.open("./log/" + program + ".csv", std::ofstream::app);

	myfile << "\"" + program + "\"";
	myfile << "; " + to_string(threads);
	myfile << "; " + to_string(seconds_runningTime);
	myfile << "\n";

	myfile.close();
}




