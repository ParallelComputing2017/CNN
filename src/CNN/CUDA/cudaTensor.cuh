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
	tensor_t<float> *h_tensor, *d_tensor;
	float *d_tensor_data;
	int tensor_data_size;

public:

	__device__ static float& get(tensor_t<float> *t, int _x, int _y, int _z) {
		assert(_x >= 0 && _y >= 0 && _z >= 0);
		assert(_x < t->size.x && _y < t->size.y && _z < t->size.z);

		return t->data[_z * (t->size.x * t->size.y) + _y * (t->size.x) + _x];
	}

	__device__ static void set(tensor_t<float> *t, int _x, int _y, int _z,
			float value) {
		assert(_x >= 0 && _y >= 0 && _z >= 0);
		assert(_x < t->size.x && _y < t->size.y && _z < t->size.z);

		t->data[_z * (t->size.x * t->size.y) + _y * (t->size.x) + _x] = value;
	}

	cudaTensor(tensor_t<float> *tensor) :
			h_tensor(tensor), tensor_data_size(0), d_tensor(NULL), d_tensor_data(
					NULL) {
	}

	void hostToDevice() {

		int tensor_mem_size = sizeof(*h_tensor);

		if (h_tensor == NULL) {
			fprintf(stderr, "Failed to allocate host vectors!\n");
			exit (EXIT_FAILURE);
		}

		cudaMalloc((void **) &d_tensor, tensor_mem_size);
		cudaCheckError()

		cudaMemcpy(d_tensor, h_tensor, tensor_mem_size, cudaMemcpyHostToDevice);
		cudaCheckError()

			// DATA
		tensor_data_size = sizeof(float) * h_tensor->getSize().x
				* h_tensor->getSize().y * h_tensor->getSize().z;

		cudaMalloc((void **) &d_tensor_data, tensor_data_size);
		cudaCheckError()

			// Copy data to device
		cudaMemcpy(d_tensor_data, h_tensor->data, tensor_data_size,
				cudaMemcpyHostToDevice);
		cudaCheckError()

			// Copy pointer
		cudaMemcpy(&(d_tensor->data), &d_tensor_data, sizeof(d_tensor->data),
				cudaMemcpyHostToDevice);
		cudaCheckError()

	}

	tensor_t<float>* devicePointer() {
		return d_tensor;
	}
	tensor_t<float>* hostPointer() {
		return h_tensor;
	}

	void deviceToHost() {
		cudaMemcpy(h_tensor->data, d_tensor_data, tensor_data_size,
				cudaMemcpyDeviceToHost);
		cudaCheckError()

	}

	void deviceFree() {
		cudaFree(d_tensor);
		cudaCheckError()

	}
};
