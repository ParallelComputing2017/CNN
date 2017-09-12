#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <omp.h>

#include "byteswap.h"
#include "cnn.h"

using namespace std;

#pragma pack(push, 1)
struct RGB {
	uint8_t r, g, b;
};
#pragma pack(pop)

float train(vector<layer_t*>& layers, tensor_t<float>& data,
		tensor_t<float>& expected) {
	for (int i = 0; i < layers.size(); i++) {
		if (i == 0) {
			activate(layers[i], data);
		}
		else
			activate(layers[i], layers[i - 1]->out);
	}

	tensor_t<float> grads = layers.back()->out - expected;

	for (int i = layers.size() - 1; i >= 0; i--) {
		if (i == layers.size() - 1)
			calc_grads(layers[i], grads);
		else
			calc_grads(layers[i], layers[i + 1]->grads_in);
	}

	for (int i = 0; i < layers.size(); i++) {
		fix_weights(layers[i]);
	}

	float err = 0;
	for (int i = 0; i < grads.size.x * grads.size.y * grads.size.z; i++) {
		float f = expected.data[i];
		if (f > 0.5)
			err += abs(grads.data[i]);
	}
	return err * 100;
}

void forward(vector<layer_t*>& layers, tensor_t<float>& data) {
	for (int i = 0; i < layers.size(); i++) {
		if (i == 0)
			activate(layers[i], data);
		else
			activate(layers[i], layers[i - 1]->out);
	}
}

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

/**
 * Get an example layers.
 */
vector<layer_t*> getExampleLayers1(vector<case_t> cases) {

	vector<layer_t*> layers;

	conv_layer_t * layer1 = new conv_layer_t(1, 5, 8, cases[0].data.size); // 28 * 28 * 1 -> 24 * 24 * 8
	relu_layer_t * layer2 = new relu_layer_t(layer1->out.size);
	pool_layer_t * layer3 = new pool_layer_t(2, 2, layer2->out.size); // 24 * 24 * 8 -> 12 * 12 * 8
	fc_layer_t * layer4 = new fc_layer_t(layer3->out.size, 10);	// 4 * 4 * 16 -> 10

	layers.push_back((layer_t*) layer1);
	layers.push_back((layer_t*) layer2);
	layers.push_back((layer_t*) layer3);
	layers.push_back((layer_t*) layer4);

	return layers;
}

/**
 * Get an example layers.
 */
vector<layer_t*> getExampleLayers2(vector<case_t> cases) {

	vector<layer_t*> layers;

	conv_layer_t * layer1 = new conv_layer_t(1, 5, 8, cases[0].data.size);// 28 * 28 * 1 -> 24 * 24 * 8
	relu_layer_t * layer2 = new relu_layer_t(layer1->out.size);
	pool_layer_t * layer3 = new pool_layer_t(2, 2, layer2->out.size);// 24 * 24 * 8 -> 12 * 12 * 8

	conv_layer_t * layer4 = new conv_layer_t(1, 3, 10, layer3->out.size);// 12 * 12 * 6 -> 10 * 10 * 10
	relu_layer_t * layer5 = new relu_layer_t(layer4->out.size);
	pool_layer_t * layer6 = new pool_layer_t(2, 2, layer5->out.size);// 10 * 10 * 10 -> 5 * 5 * 10

	fc_layer_t * layer7 = new fc_layer_t(layer6->out.size, 10);	// 4 * 4 * 16 -> 10

	layers.push_back((layer_t*) layer1);
	layers.push_back((layer_t*) layer2);
	layers.push_back((layer_t*) layer3);
	layers.push_back((layer_t*) layer4);
	layers.push_back((layer_t*) layer5);
	layers.push_back((layer_t*) layer6);
	layers.push_back((layer_t*) layer7);

	return layers;
}

int mainExample() {

	vector<case_t> cases = read_test_cases();

	vector<layer_t*> layers;

	layers = getExampleLayers1(cases);

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

		forward(layers, image);
		tensor_t<float>& out = layers.back()->out;

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

int openMP(int numThreads) {

	vector<case_t> cases = read_test_cases();

	vector<layer_t*> master;

	master = getExampleLayers1(cases);

	//layers = getExampleLayers2(cases);

	printf("Training cases: %i \n", cases.size());

	vector<layer_t*> slaves[numThreads];

	for (int t = 0; t < numThreads; t++) {
		slaves[t] = getExampleLayers1(cases);
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

				if (ep % 2000 == 0) {
					printf("thread: %i,\t ep: %i,\t i: %i,\t err: %f \n",
							threadId,
							ep, i, amse / ic);
				}
			}
			break;
		}
	}
	// end:
	master = slaves[0];

	// Join slaves

	for (int l = 0; l < master.size(); l++) {

		layer_t* masterLayer = master.at(l);

		switch (masterLayer->type) {
		case layer_type::conv:
			((conv_layer_t*) masterLayer);
			break;
		case layer_type::relu:
			((relu_layer_t*) masterLayer);
			break;
		case layer_type::fc: {
			// TODO remove
			printf("*** Layer %i \n", l);

			tensor_t<float> weights = ((fc_layer_t*) masterLayer)->weights;

			for (vector<layer_t*> slave : slaves) {
				layer_t* slaveLayer = slave[l];
				if (slaveLayer->type != layer_type::fc) {
					printf("ERROR Layer type");
				}
				weights = weights + ((fc_layer_t*) slaveLayer)->weights;
			}
			weights = weights / ((float) sizeof(slaves) / sizeof(slaves[0]));

			// TODO remove
			printf("new weights %i, sizeof(slaves): %f \n", weights.size,
					((float) sizeof(slaves) / sizeof(slaves[0])));
			print_tensor(weights);

			((fc_layer_t*) masterLayer)->setWeights(weights);

			print_tensor(((fc_layer_t*) masterLayer)->weights);
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
