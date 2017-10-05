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

using namespace std;

struct case_t {
	tensor_t<float> data;
	tensor_t<float> out;
};

uint8_t* read_file(const char* szFile) {
	ifstream file(szFile, ios::binary | ios::ate);
	streamsize size = file.tellg();
	file.seekg(0, ios::beg);

	if (size == -1)
		return nullptr;

	uint8_t* buffer = new uint8_t[size];
	file.read((char*) buffer, size);
	return buffer;
}

vector<case_t> read_test_cases() {
	vector<case_t> cases;

	uint8_t* train_image = read_file("data/mnist/train-images.idx3-ubyte");
	uint8_t* train_labels = read_file("data/mnist/train-labels.idx1-ubyte");

	uint32_t case_count = byteswap_uint32(*(uint32_t*) (train_image + 4));

	for (int i = 0; i < case_count; i++) {
		case_t c { tensor_t<float>(28, 28, 1), tensor_t<float>(10, 1, 1) };

		uint8_t* img = train_image + 16 + i * (28 * 28);
		uint8_t* label = train_labels + 8 + i;

		for (int x = 0; x < 28; x++)
			for (int y = 0; y < 28; y++)
				c.data(x, y, 0) = img[x + y * 28] / 255.f;

		for (int b = 0; b < 10; b++)
			c.out(b, 0, 0) = *label == b ? 1.0f : 0.0f;

		cases.push_back(c);
	}
	delete[] train_image;
	delete[] train_labels;

	return cases;
}

int singleTest(vector<layer_t*> master) {
	uint8_t * data = read_file("data/test.ppm");

	int digit = -1;

	if (data) {
		uint8_t * usable = data;

		while (*(uint32_t*) usable != 0x0A353532)
			usable++;

		RGB * rgb = (RGB*) usable;

		tensor_t<float> image(28, 28, 1);
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				RGB rgb_ij = rgb[i * 28 + j];
				image(j, i, 0) = (((float) rgb_ij.r + rgb_ij.g + rgb_ij.b)
						/ (3.0f * 255.f));
			}
		}

		forward(master, image);
		tensor_t<float>& out = master.back()->out;

		float maxProbability = 0.0;

		for (int i = 0; i < 10; i++) {
			float probability = out(i, 0, 0) * 100.0f;
			if (probability > maxProbability) {
				digit = i;
				maxProbability = probability;
			}
			printf("[%i] %f\n", i, probability);
		}

		delete[] data;
	}

	return digit;
}


int mainExample() {

	vector<case_t> cases = read_test_cases();

	vector<layer_t*> layers;

	layers = getExampleLayers1(cases[0].data.size);

	//layers = getExampleLayers2(cases);

	float amse = 0;
	int ic = 0;

	printf("Training cases: %i \n", cases.size());

	for (long ep = 0; ep < cases.size();) {

		for (case_t& t : cases) {
			float xerr = train(layers, t.data, t.out);
			amse += xerr;

			ep++;
			ic++;

			if (ep % 1000 == 0) {
				cout << "case " << ep << " err=" << amse / ic << endl;
			}
		}
	}
	// end:

	// TEST

	return singleTest(layers);
}

vector<layer_t*> training(vector<case_t> cases, int batchStart, int batchEnd,
		vector<layer_t*> layers) {

	float amse = 0;
	int ic = 0;

	for (long ep = 0; ep < 1;) {

		for (int i = batchStart; i < batchEnd; i++) {
			case_t& t = cases.at(i);
			float xerr = train(layers, t.data, t.out);
			amse += xerr;

			ep++;
			ic++;

			if (ep % 4000 == 0) {
				printf("ep: %i,\t i: %i,\t err: %f \n",
						ep, i, amse / ic);
			}
		}
	}

	return layers;
}

int openMP(int numThreads) {

	vector<case_t> cases = read_test_cases();

	vector<layer_t*> master;

	master = getExampleLayers1(cases[0].data.size);

	//layers = getExampleLayers2(cases);

	printf("Training cases: %i \n", cases.size());

	vector<vector<layer_t*>> slaves;

	for (int t = 0; t < numThreads; t++) {
		slaves.push_back(getExampleLayers1(cases[0].data.size));
	}

#pragma omp parallel num_threads(numThreads)
	{
		int threadId = omp_get_thread_num();

		int batchSize = cases.size() / numThreads;
		int batchStart = batchSize * threadId;
		int batchEnd = batchStart + batchSize - 1;

		printf("thread: %i, batchSize: %i, batchStart: %i, batchEnd: %i,  \n",
				threadId, batchSize, batchStart, batchEnd);

		vector<layer_t*> layers = slaves[threadId];

		float amse = 0;
		int ic = 0;

		for (long ep = 0; ep < batchSize;) {

			for (int i = batchStart; i < batchEnd; i++) {
				case_t& t = cases.at(i);
				float xerr = train(layers, t.data, t.out);
				amse += xerr;

				ep++;
				ic++;

				if (ep % 4000 == 0) {
					printf("thread: %i,\t ep: %i,\t i: %i,\t err: %f \n",
							threadId, ep, i, amse / ic);
				}
			}
			break;
		}
	}
	// end:

	// warm the model
	master = getExampleLayers1(cases[0].data.size);

	printf("*** init master *** \n");
	singleTest(master);

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
			printf("*** Layer %i \n", layer);

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
			printf("sizeof(slaves): %f \n",
					((float) sizeof(slaves) / sizeof(slaves[0])));

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

	// TODO remove
	printf("*** END OF TRAINING *** \n");

	// TEST
	printf("*** Slave 0 *** \n");
	singleTest(slaves[0]);

	printf("*** Master *** \n");
	return singleTest(master);
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

	vector<case_t> cases = read_test_cases();

	vector<layer_t*> master;

	master = getExampleLayers1(cases[0].data.size);

	//layers = getExampleLayers2(cases);

	printf("Training cases: %i \n", cases.size());

	vector<layer_t*> slaves[numThreads];

	thread_data thread_data_array[numThreads];
	pthread_t threads[numThreads];
	int threadId[numThreads], i, *retval;

	for (i = 0; i < numThreads; i++) {
		thread_data_array[i].thread_id = i;
		thread_data_array[i].numThreads = numThreads;
		thread_data_array[i].cases = cases;
		slaves[i] = getExampleLayers1(cases[0].data.size);
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
			printf("*** Layer %i \n", layer);

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
			printf("sizeof(slaves): %f \n",
					((float) sizeof(slaves) / sizeof(slaves[0])));

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



// TODO remove
	printf("*** END OF TRAINING *** \n");

	return singleTest(master);

}


