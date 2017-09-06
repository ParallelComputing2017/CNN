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

#include "simple_cnn/Example_MNIST/example.h"

void writeCSV(string program, int threads, float runningTime);
void printLog(string mode, cpu_timer timer, double result);

int main(int argc, char *argv[]) {

	int threads = 4;
	string self(argv[0]);
	string mode = "all";

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
		timer.stop();

		printLog("OpenMP", timer, 0);

	}
	if (mode.compare("single") == 0) {

		timer.start();
		double pi = mainExample();
		timer.stop();

		printLog("Single", timer, pi);
	}

	return 0;

}

void printLog(string mode, cpu_timer timer, double result) {

	printf("%s\t time: %s\t result: %f \n", mode.c_str(),
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




