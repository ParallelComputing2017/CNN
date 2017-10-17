#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>

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
	for (unsigned i = 0; i < layers.size(); i++) {
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


/**
 * Get an example layers.
 */
vector<layer_t*> getExampleLayers1(tdsize inputSize) {

	vector<layer_t*> layers;

	conv_layer_t * layer1 = new conv_layer_t(1, 5, 8, inputSize); // 28 * 28 * 1 -> 24 * 24 * 8
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
vector<layer_t*> getExampleLayers1Cuda(tdsize inputSize) {

	vector<layer_t*> layers;

	conv_layer_t * layer1 = new conv_layer_t(1, 5, 8, inputSize); // 28 * 28 * 1 -> 24 * 24 * 8
	relu_layer_t * layer2 = new relu_layer_t(layer1->out.size);
	pool_layer_t * layer3 = new pool_layer_t(2, 2, layer2->out.size); // 24 * 24 * 8 -> 12 * 12 * 8
	fc_layer_cuda_t * layer4 = new fc_layer_cuda_t(layer3->out.size, 10);	// 4 * 4 * 16 -> 10

	layers.push_back((layer_t*) layer1);
	layers.push_back((layer_t*) layer2);
	layers.push_back((layer_t*) layer3);
	layers.push_back((layer_t*) layer4);

	return layers;
}

/**
 * Get an example layers.
 */
vector<layer_t*> getExampleLayers2(tdsize inputSize) {

	vector<layer_t*> layers;

	conv_layer_t * layer1 = new conv_layer_t(1, 5, 8, inputSize);// 28 * 28 * 1 -> 24 * 24 * 8
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

