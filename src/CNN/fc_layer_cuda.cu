#include <stdio.h>
#include <math.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <vector>
#include <algorithm>   // std::min

#include <math.h>
#include <float.h>
#include <string.h>

#include "gradient_t.h"
#include "layer_t.h"
#include "optimization_method.h"

#include "fc_layer_cuda.h"

#include "CUDA/utils.cuh"

void cudaActivate(tensor_t<float> in, tensor_t<float> weights,
		std::vector<float> &input, tensor_t<float> &out);

__host__ __device__ float activator_function(float x) {
	//return tanhf( x );
	float sig = 1.0f / (1.0f + exp(-x));
	return sig;
}

__host__ void fc_layer_cuda_t::activate(tensor_t<float>& in) {
	this->in = in;
	//activate();

	cudaActivate(this->in, this->weights, this->input, this->out);

	// TODO
	exit(EXIT_SUCCESS);

}

/**
 * CUDA Kernel Device code
 *
 */
/*****************************************************************************/

__device__ float& get(tensor_t<float> *t, int _x, int _y, int _z) {
	assert(_x >= 0 && _y >= 0 && _z >= 0);
	assert(_x < t->size.x && _y < t->size.y && _z < t->size.z);

	return t->data[_z * (t->size.x * t->size.y) + _y * (t->size.x) + _x];
}

__device__ void set(tensor_t<float> *t, int _x, int _y, int _z, float value) {
	assert(_x >= 0 && _y >= 0 && _z >= 0);
	assert(_x < t->size.x && _y < t->size.y && _z < t->size.z);

	t->data[_z * (t->size.x * t->size.y) + _y * (t->size.x) + _x] = value;
}

__global__ void activate_cuda(tensor_t<float> *d_in, tensor_t<float> *d_weights,
		float *d_input, tensor_t<float> *d_out) {

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	int i = ((int) index / (d_in->size.y * d_in->size.z)) % d_in->size.x;
	int j = ((int) index / d_in->size.z) % d_in->size.y;
	int k = index % d_in->size.z;

	printf("index: %i  (x, y, z)=(%i, %i, %i)  (i, j, k)=(%i, %i, %i) \n",
	 index, d_in->size.x, d_in->size.y, d_in->size.z, i, j, k);

	for (int n = 0; n < d_out->size.x; n++) {
		float inputv = 0;

		// map
		int m = k * (d_in->size.x * d_in->size.y) + j * (d_in->size.x) + i;

		inputv += get(d_in, i, j, k) * get(d_weights, m, n, 0);

		//printf("inputv: %f \n", inputv);

		*(d_input + n) = inputv;

		set(d_out, n, 0, 0, activator_function(inputv));
	}

}

/******************************************************************************
 * Host main routine
 */
void cudaActivate(tensor_t<float> in, tensor_t<float> weights,
		std::vector<float> &input, tensor_t<float> &out) {

	int blocksPerGrid, threadsPerBlock;
	int totalThreads;
	tensor_t<float> *h_in, *d_in;
	tensor_t<float> *h_weights, *d_weights;
	float *h_input, *d_input;
	tensor_t<float> *h_out, *d_out;

	// Get device info
	int dev = 0;
	cudaSetDevice(dev);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

	int requiredThreads = in.size.x * in.size.y * in.size.z;

	blocksPerGrid = std::min(deviceProp.multiProcessorCount, 2);

	threadsPerBlock = std::min(deviceProp.maxThreadsPerBlock,
			requiredThreads / blocksPerGrid);

	totalThreads = blocksPerGrid * threadsPerBlock;

	h_in = &in;
	int in_mem_size = sizeof(in);

	if (h_in == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	cudaMalloc((void **) &d_in, in_mem_size);
	cudaCheckError("cudaMalloc IN tensor");

	cudaMemcpy(d_in, h_in, in_mem_size, cudaMemcpyHostToDevice);
	cudaCheckError("cudaMemcopy to device IN tensor");

	// IN DATA

	float *d_in_data;
	long in_data_size = sizeof(*d_in_data) * in.getSize().x * in.getSize().y
			* in.getSize().z;

	//printf("sizeof(in)= %lu , in_data_size = %lu \n", sizeof(in), in_data_size);

	cudaMalloc((void **) &d_in_data, in_data_size);
	cudaCheckError("cudaMalloc IN tensor data");

	cudaMemcpy(d_in_data, in.data, in_data_size, cudaMemcpyHostToDevice);
	cudaCheckError("cudaMemcpy to device IN tensor data");

	cudaMemcpy(&(d_in->data), &d_in_data, sizeof(d_in->data),
			cudaMemcpyHostToDevice);
	cudaCheckError("cudaMemcpy Binding pointers of IN tensor data");

	// Copy weights

	h_weights = &weights;
	long weights_mem_size = sizeof(weights);

	//printf("sizeof(weights) == weights_mem_size => %lu == %lu \n",sizeof(weights), weights_mem_size);

	if (h_weights == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	cudaMalloc((void **) &d_weights, weights_mem_size);
	cudaCheckError("cudaMalloc Weights tensor");

	cudaMemcpy(d_weights, h_weights, weights_mem_size, cudaMemcpyHostToDevice);
	cudaCheckError("cudaMemcpy to device Weights tensor");

	// Weights DATA

	float *d_weights_data;
	long weights_data_size = sizeof(*d_weights_data) * weights.getSize().x
			* weights.getSize().y * weights.getSize().z;

	cudaMalloc((void **) &d_weights_data, weights_data_size);
	cudaCheckError("cudaMalloc Weights tensor data");

	cudaMemcpy(d_weights_data, weights.data, weights_data_size,
			cudaMemcpyHostToDevice);
	cudaCheckError("cudaMemcpy to device Weights tensor data");

	cudaMemcpy(&(d_weights->data), &d_weights_data, sizeof(d_weights->data),
			cudaMemcpyHostToDevice);
	cudaCheckError("cudaMemcpy Binding pointers of Weights tensor data");

	// Reserve input memory space

	h_input = &input[0];
	long input_mem_size = sizeof(input[0]) * input.size();

	//printf("input_mem_size : %lu \n", input_mem_size);

	cudaMalloc((void **) &d_input, input_mem_size);
	cudaCheckError("cudaMalloc Input array");

	cudaMemcpy(d_input, h_input, input_mem_size, cudaMemcpyHostToDevice);
	cudaCheckError("cudaMemcpy to device Input array");

	// Reserve memory space for OUT

	h_out = &out;
	int out_mem_size = sizeof(out);

	cudaMalloc((void **) &d_out, out_mem_size);
	cudaCheckError("cudaMalloc Out tensor");

	cudaMemcpy(d_out, h_out, out_mem_size, cudaMemcpyHostToDevice);
	cudaCheckError("cudaMemcpy to device Out tensor");

	// Out DATA

	float *d_out_data;
	long out_data_size = sizeof(*d_out_data) * h_out->getSize().x
			* h_out->getSize().y * h_out->getSize().z;

	//printf("out_data_size : %lu \n", out_data_size);

	cudaMalloc((void **) &d_out_data, out_data_size);
	cudaCheckError("cudaMalloc Out tensor data");

	cudaMemcpy(d_out_data, h_out->data, out_data_size, cudaMemcpyHostToDevice);
	cudaCheckError("cudaMemcpy to device Out tensor data");

	cudaMemcpy(&(d_out->data), &d_out_data, sizeof(d_out->data),
			cudaMemcpyHostToDevice);
	cudaCheckError("cudaMemcpy Binding pointers of Out tensor data");

	// Lanzar KERNEL

	Logger::debug("CUDA kernel launch with %d blocks of %d threads. Total: %i\n",
			blocksPerGrid, threadsPerBlock, totalThreads);

	activate_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_weights, d_input,
			d_out);

	cudaDeviceSynchronize();

	cudaCheckError("Launch kernel");

	// get input array

	cudaMemcpy(h_input, d_input, input_mem_size, cudaMemcpyDeviceToHost);
	cudaCheckError("cudaMemcpy to host Input array");

	cudaFree(d_input);
	cudaCheckError("cudaFree Input array");

	// Get Out DATA

	cudaMemcpy(h_out->data, d_out_data, out_data_size, cudaMemcpyDeviceToHost);
	cudaCheckError("cudaMemcpy to host Out tensor data");

	cudaFree(d_out);
	cudaCheckError("cudaFree Out tensor");

	// Free host memory

	cudaDeviceReset();

	cudaCheckError("cudaDeviceReset");
	// TODO remove
	//printf("cuda out: %i, %i, %i \n", h_out->getSize().x, h_out->getSize().y,h_out->getSize().z);
	//printf("cuda out[0,0,0]: %f, \n", h_out->get(0, 0, 0));
	//print_tensor(*h_out);
}

