/*
 * cudaTensor.cuh
 *
 *  Created on: 7/11/2017
 *      Author: sebas
 */

#pragma once

#include "../tensor_t.h"
#include "utils.cuh"

class cudaTensor {

private:
	tensor_t<float> *tensor;
	tensor_t<float> *h_in, *d_in;
	float *d_in_data;
	int in_data_size;

public:

	__device__ static float& get(tensor_t<float> *t, int _x, int _y, int _z) {
		assert(_x >= 0 && _y >= 0 && _z >= 0);
		assert(_x < t->size.x && _y < t->size.y && _z < t->size.z);

		return t->data[_z * (t->size.x * t->size.y) + _y * (t->size.x) + _x];
	}

	__device__ static void set(tensor_t<float> *t, int _x, int _y, int _z, float value) {
		assert(_x >= 0 && _y >= 0 && _z >= 0);
		assert(_x < t->size.x && _y < t->size.y && _z < t->size.z);

		t->data[_z * (t->size.x * t->size.y) + _y * (t->size.x) + _x] = value;
	}

	cudaTensor(tensor_t<float> *tensor) :
			tensor(tensor), in_data_size(0) {
	}

	void hostToDevice() {
		h_in = tensor;

		int in_mem_size = sizeof(*tensor);

		if (h_in == NULL) {
			fprintf(stderr, "Failed to allocate host vectors!\n");
			exit (EXIT_FAILURE);
		}

		cudaMalloc((void **) &d_in, in_mem_size);
		cudaCheckError();

		cudaMemcpy(d_in, h_in, in_mem_size, cudaMemcpyHostToDevice);
		cudaCheckError();

		// DATA
		in_data_size = sizeof(*d_in_data) * tensor->getSize().x
				* tensor->getSize().y * tensor->getSize().z;

		cudaMalloc((void **) &d_in_data, in_data_size);
		cudaCheckError();

		cudaMemcpy(d_in_data, tensor->data, in_data_size,
				cudaMemcpyHostToDevice);
		cudaCheckError();

		cudaMemcpy(&(d_in->data), &d_in_data, sizeof(d_in->data),
				cudaMemcpyHostToDevice);
		cudaCheckError();
	}

	tensor_t<float>* devicePointer(){
		return d_in;
	}

	void deviceToHost() {
		cudaMemcpy(h_in->data, d_in_data, in_data_size, cudaMemcpyDeviceToHost);
		cudaCheckError();
	}

	void deviceFree() {
		cudaFree(d_in);
		cudaCheckError();
	}
};
