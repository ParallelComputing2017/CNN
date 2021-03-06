#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "byteswap.h"

#include "relu_layer_t.h"
#include "pool_layer_t.h"
#include "fc_layer.h"
#include "dropout_layer_t.h"

#include "CUDA/cudaConvLayer.h"

#include "OpenCL/openCLConvLayer.h"

using namespace std;

class NeuralNetwork {

private:
	vector<layer_t*> layers;

	void forward(tensor_t<float>& data) {
		for (int i = 0; i < layers.size(); i++) {
			layer_t* layer = layers[i];
			if (i == 0) {
				layer->activate(data);
			} else {
				layer->activate(layers[i - 1]->out);
			}
		}
	}

public:

	NeuralNetwork(vector<layer_t*>& layers) :
			layers(layers) {
	}

	void test(tensor_t<float>& data) {
		forward(data);
	}

	float train(tensor_t<float>& data, tensor_t<float>& expected) {

		forward(data);

		tensor_t<float> grads = layers.back()->out - expected;

		for (int i = layers.size() - 1; i >= 0; i--) {
			layer_t* layer = layers[i];
			if (i == layers.size() - 1) {
				layer->calc_grads(grads);
			} else {
				layer->calc_grads(layers[i + 1]->grads_in);
			}
		}

		for (int i = 0; i < layers.size(); i++) {
			layer_t* layer = layers[i];
			layer->fix_weights();
		}

		float err = 0;
		int gradsSize = grads.getSize().x * grads.getSize().y
				* grads.getSize().z;

		for (int i = 0; i < gradsSize; i++) {
			float f = expected.data[i];
			if (f > 0.5)
				err += abs(grads.data[i]);
		}
		return err * 100;
	}
};

/**
 * Get an example layers.
 */
vector<layer_t*> getExampleLayers1(tdsize inputSize) {

	vector<layer_t*> layers;

	conv_layer_t * layer1 = new conv_layer_t(1, 5, 8, inputSize); // 28 * 28 * 1 -> 24 * 24 * 8
	relu_layer_t * layer2 = new relu_layer_t(layer1->out.getSize());
	pool_layer_t * layer3 = new pool_layer_t(2, 2, layer2->out.getSize()); // 24 * 24 * 8 -> 12 * 12 * 8
	fc_layer_t * layer4 = new fc_layer_t(layer3->out.getSize(), 10);// 4 * 4 * 16 -> 10

	layers.push_back((layer_t*) layer1);
	layers.push_back((layer_t*) layer2);
	layers.push_back((layer_t*) layer3);
	layers.push_back((layer_t*) layer4);

	return layers;
}

/**
 * Get an example layers.
 */
vector<layer_t*> getExampleCuda(tdsize inputSize) {

	vector<layer_t*> layers;

	layer_t * convLayer = new CudaConvLayer(1, 5, 8, inputSize); // 28 * 28 * 1 -> 24 * 24 * 8
	relu_layer_t * layer2 = new relu_layer_t(convLayer->out.getSize());
	pool_layer_t * layer3 = new pool_layer_t(2, 2, layer2->out.getSize()); // 24 * 24 * 8 -> 12 * 12 * 8
	layer_t * fcLayer = new fc_layer_t(layer3->out.getSize(), 10);// 4 * 4 * 16 -> 10

	layers.push_back((layer_t*) convLayer);
	layers.push_back((layer_t*) layer2);
	layers.push_back((layer_t*) layer3);
	layers.push_back((layer_t*) fcLayer);

	return layers;
}

/**
 * Get an example layers.
 */
vector<layer_t*> getExampleOpenCL(tdsize inputSize) {

	vector<layer_t*> layers;

	layer_t * convLayer = new OpenCLConvLayer(1, 5, 8, inputSize); // 28 * 28 * 1 -> 24 * 24 * 8
	relu_layer_t * layer2 = new relu_layer_t(convLayer->out.getSize());
	pool_layer_t * layer3 = new pool_layer_t(2, 2, layer2->out.getSize()); // 24 * 24 * 8 -> 12 * 12 * 8
	layer_t * fcLayer = new fc_layer_t(layer3->out.getSize(), 10);// 4 * 4 * 16 -> 10

	layers.push_back((layer_t*) convLayer);
	layers.push_back((layer_t*) layer2);
	layers.push_back((layer_t*) layer3);
	layers.push_back((layer_t*) fcLayer);

	return layers;
}

/**
 * Get an example layers.
 */
vector<layer_t*> getExampleLayers2(tdsize inputSize) {

	vector<layer_t*> layers;

	conv_layer_t * layer1 = new conv_layer_t(1, 5, 8, inputSize);// 28 * 28 * 1 -> 24 * 24 * 8
	relu_layer_t * layer2 = new relu_layer_t(layer1->out.getSize());
	pool_layer_t * layer3 = new pool_layer_t(2, 2, layer2->out.getSize());// 24 * 24 * 8 -> 12 * 12 * 8

	conv_layer_t * layer4 = new conv_layer_t(1, 3, 10, layer3->out.getSize());// 12 * 12 * 6 -> 10 * 10 * 10
	relu_layer_t * layer5 = new relu_layer_t(layer4->out.getSize());
	pool_layer_t * layer6 = new pool_layer_t(2, 2, layer5->out.getSize());// 10 * 10 * 10 -> 5 * 5 * 10

	fc_layer_t * layer7 = new fc_layer_t(layer6->out.getSize(), 10);// 4 * 4 * 16 -> 10

	layers.push_back((layer_t*) layer1);
	layers.push_back((layer_t*) layer2);
	layers.push_back((layer_t*) layer3);
	layers.push_back((layer_t*) layer4);
	layers.push_back((layer_t*) layer5);
	layers.push_back((layer_t*) layer6);
	layers.push_back((layer_t*) layer7);

	return layers;
}
