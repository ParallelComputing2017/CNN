#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <pthread.h>
#include <omp.h>

#include "CNN/neural_network.h"
#include "mnist.h"

using namespace std;

vector<layer_t*> training(vector<case_t> cases, int batchStart, int batchEnd,
		vector<layer_t*> layers) {

	Logger::info("Start training \n");

	float amse = 0;
	int ic = 0;

	for (long ep = 0; ep < 1;) {

		for (int i = batchStart; i < batchEnd; i++) {
			case_t& t = cases.at(i);
			float xerr = train(layers, t.data, t.out);
			amse += xerr;

			ep++;
			ic++;

			if (ep % 50 == 0) {

				Logger::info("ep: %lu,\t i: %i,\t err: %f \n", ep, i, amse / ic);
			}
		}
	}

	return layers;
}

int sequential() {

	vector<case_t> cases = read_training_cases();

	vector<layer_t*> layers = getExampleLayers1(cases[0].data.getSize());

	layers = training(cases, 0, cases.size() - 1, layers);

	// TEST
	return singleTest(layers);
}

/*
 * Join the slaves models with the master
 * */
vector<layer_t*> joinSlaves(vector<layer_t*> master,
		vector<vector<layer_t*>> slaves) {

	// Join slaves
	for (int layer = 0; layer < master.size(); layer++) {

		layer_t* masterLayer = master.at(layer);

		switch (masterLayer->type) {
		case layer_type::conv:
			((conv_layer_t*) masterLayer);
			break;
		case layer_type::relu:
			((relu_layer_t*) masterLayer);
			break;
		case layer_type::fc: {
			// TODO remove
			//printf("*** Layer %i \n", layer);

			fc_layer_t* fcMasterLayer = (fc_layer_t*) (masterLayer);

			for (vector<layer_t*> slave : slaves) {
				layer_t* slaveLayer = slave[layer];
				if (slaveLayer->type != layer_type::fc) {
					printf("ERROR Layer type");
				}
				fc_layer_t* fcSlaveLayer = (fc_layer_t*) (slaveLayer);

				fcMasterLayer->updateWeights(fcSlaveLayer->weights);
			}

			// TODO remove
			/*printf("sizeof(slaves): %f \n",
			 ((float) sizeof(slaves) / sizeof(slaves[0])));*/

			break;
		}
		case layer_type::pool:

			((pool_layer_t*) masterLayer);
			break;
		case layer_type::dropout_layer:
			((dropout_layer_t*) masterLayer);
			break;
		default:
			assert(false);
			break;
		}
	}

	return master;
}

int openMP(int numThreads) {

	vector<case_t> cases = read_training_cases();

	vector<layer_t*> master;

	master = getExampleLayers1(cases[0].data.getSize());

	//layers = getExampleLayers2(cases);

	printf("Training cases: %lu \n", cases.size());

	vector<vector<layer_t*>> slaves;

	for (int t = 0; t < numThreads; t++) {
		slaves.push_back(getExampleLayers1(cases[0].data.getSize()));
	}

#pragma omp parallel num_threads(numThreads)
	{
		int threadId = omp_get_thread_num();

		int batchSize = cases.size() / numThreads;
		int batchStart = batchSize * threadId;
		int batchEnd = batchStart + batchSize - 1;

		// TODO remove
		/*printf("thread: %i, batchSize: %i, batchStart: %i, batchEnd: %i,  \n",
		 threadId, batchSize, batchStart, batchEnd);*/

		vector<layer_t*> layers = slaves[threadId];

		layers = training(cases, batchStart, batchEnd, layers);

	}
	// end:

	// warm the model
	master = getExampleLayers1(cases[0].data.getSize());

	// TODO remove
	/*printf("*** init master *** \n");
	 singleTest(master);*/

	// Join slaves
	//master = joinSlaves(master, slaves);
	// TODO
	master = slaves[0];

	// TODO remove
	printf("*** END OF TRAINING *** \n");

	fullTest(master);

	return singleTest(master);
}

/*
 * Run in a Nvidia GPU
 */
int cuda(int maxBlocks) {

	vector<case_t> cases = read_training_cases();



	vector<layer_t*> layers = getExampleCuda(cases[0].data.getSize());

	int casesSize = cases.size() - 1-59000;

	printf("Training cases: %i \n", casesSize);

	layers = training(cases, 0, casesSize, layers);

	// TEST
	//fullTest(layers);

	return singleTest(layers);
}

struct thread_data {
	int thread_id;
	int numThreads;
	vector<case_t> cases;
	vector<layer_t*> slaves;
};

void *training(void *threadarg) {

	thread_data my_data = *(thread_data *) threadarg;

	int thread_id = my_data.thread_id;
	vector<case_t> cases = my_data.cases;
	int batchSize = cases.size() / my_data.numThreads;
	int batchStart = batchSize * thread_id;
	int batchEnd = batchStart + batchSize - 1;
	vector<layer_t*> layers = my_data.slaves;

	training(cases, batchStart, batchEnd, layers);

}

int posix(int numThreads) {

	vector<case_t> cases = read_training_cases();

	vector<layer_t*> master;

	master = getExampleLayers1(cases[0].data.getSize());

	//layers = getExampleLayers2(cases);

	printf("Training cases: %lu \n", cases.size());

	vector<vector<layer_t*>> slaves;

	thread_data thread_data_array[numThreads];
	pthread_t threads[numThreads];
	int threadId[numThreads], i, *retval;

	for (i = 0; i < numThreads; i++) {
		thread_data_array[i].thread_id = i;
		thread_data_array[i].numThreads = numThreads;
		thread_data_array[i].cases = cases;
		slaves.push_back(getExampleLayers1(cases[0].data.getSize()));
		thread_data_array[i].slaves = slaves[i];

		pthread_create(&threads[i], NULL, training,
				(void *) &thread_data_array[i]);
	}
	for (i = 0; i < numThreads; i++) {
		// wait for thread termination
		pthread_join(threads[i], (void**) &retval);
	}

	for (i = 1; i < numThreads; i++) {

	}

	// warm the model
	master = slaves[0];

	master = joinSlaves(master, slaves);

// TODO remove
	printf("*** END OF TRAINING *** \n");

	return singleTest(master);

}

