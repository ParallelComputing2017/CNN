/*
 * cudaTensor.cuh
 *
 *  Created on: 7/11/2017
 *      Author: sebas
 */

#pragma once

#include "../tensor_t.h"

class tensorCudaWrapper {

private:
	tensor_t<float> *tensor;
	tensor_t<float> *h_in, *d_in;
	float *d_in_data;
	int in_data_size;

public:
	tensorCudaWrapper(tensor_t<float> *tensor) :
			tensor(tensor), in_data_size(0) {
	}

	void toGPU() {
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

	void fromGPU() {
		cudaMemcpy(h_in->data, d_in_data, in_data_size, cudaMemcpyDeviceToHost);
		cudaCheckError();
	}

	void free() {
		cudaFree(d_in);
		cudaCheckError();
	}
};
