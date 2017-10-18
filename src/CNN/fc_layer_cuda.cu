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

#include "fc_layer_cuda.cuh"

void activate2cuda(tensor_t<float> in, tensor_t<float> weights,
		std::vector<float> input, tensor_t<float> out);

fc_layer_cuda_t::fc_layer_cuda_t(tdsize in_size, int out_size) :
		in(in_size.x, in_size.y, in_size.z), out(out_size, 1, 1), grads_in(
				in_size.x, in_size.y, in_size.z), weights(
				in_size.x * in_size.y * in_size.z, out_size, 1) {
	input = std::vector<float>(out_size);
	gradients = std::vector<gradient_t>(out_size);

	int maxval = in_size.x * in_size.y * in_size.z;

	for (int i = 0; i < out_size; i++)
		for (int h = 0; h < in_size.x * in_size.y * in_size.z; h++)
			weights(h, i, 0) = 2.19722f / maxval * rand() / float(RAND_MAX);
	// 2.19722f = f^-1(0.9) => x where [1 / (1 + exp(-x) ) = 0.9]
}

__host__ __device__ float activator_function(float x) {
	//return tanhf( x );
	float sig = 1.0f / (1.0f + exp(-x));
	return sig;
}

float activator_derivative(float x) {
	//float t = tanhf( x );
	//return 1 - t * t;
	float sig = 1.0f / (1.0f + exp(-x));
	return sig * (1 - sig);
}

void fc_layer_cuda_t::activate(tensor_t<float>& in) {
	this->in = in;
	//activate();
	activate2cuda(in, weights, input, out);
}

int fc_layer_cuda_t::map(point_t d) {
	return d.z * (in.size.x * in.size.y) + d.y * (in.size.x) + d.x;
}

void fc_layer_cuda_t::activate() {
	for (int n = 0; n < out.size.x; n++) {
		float inputv = 0;

		for (int i = 0; i < in.size.x; i++)
			for (int j = 0; j < in.size.y; j++)
				for (int z = 0; z < in.size.z; z++) {
					int m = map( { i, j, z });
					inputv += in(i, j, z) * weights(m, n, 0);
				}

		input[n] = inputv;

		out(n, 0, 0) = activator_function(inputv);
	}
}

void fc_layer_cuda_t::fix_weights() {
	for (int n = 0; n < out.size.x; n++) {
		gradient_t& grad = gradients[n];
		for (int i = 0; i < in.size.x; i++)
			for (int j = 0; j < in.size.y; j++)
				for (int z = 0; z < in.size.z; z++) {
					int m = map( { i, j, z });
					float& w = weights(m, n, 0);
					w = update_weight(w, grad, in(i, j, z));
				}

		update_gradient(grad);
	}
}

void fc_layer_cuda_t::calc_grads(tensor_t<float>& grad_next_layer) {
	memset(grads_in.data, 0,
			grads_in.size.x * grads_in.size.y * grads_in.size.z
					* sizeof(float));
	for (int n = 0; n < out.size.x; n++) {
		gradient_t& grad = gradients[n];
		grad.grad = grad_next_layer(n, 0, 0) * activator_derivative(input[n]);

		for (int i = 0; i < in.size.x; i++)
			for (int j = 0; j < in.size.y; j++)
				for (int z = 0; z < in.size.z; z++) {
					int m = map( { i, j, z });
					grads_in(i, j, z) += grad.grad * weights(m, n, 0);
				}
	}
}

void fc_layer_cuda_t::setWeights(tensor_t<float> newWeights) {
	weights = newWeights;
}
void fc_layer_cuda_t::updateWeights(tensor_t<float> newWeights) {
	weights = (weights + newWeights) / 2;
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

	int index = (blockDim.x * blockIdx.x) + threadIdx.x;

	printf("Index: %i, in size: (%i, %i, %i) \t", index, d_in->size.x,
			d_in->size.y, d_in->size.z);

	for (int n = 0; n < d_out->size.x; n++) {
		float inputv = 0;

		for (int i = 0; i < d_in->size.x; i++)
			for (int j = 0; j < d_in->size.y; j++)
				for (int z = 0; z < d_in->size.z; z++) {
					// map
					int m = z * (d_in->size.x * d_in->size.y)
							+ j * (d_in->size.x) + i;

					inputv += get(d_in,i, j, z) * get(d_weights,m, n, 0);
				}

		*(d_input + n) = inputv;

		get(d_out, n, 0, 0)= activator_function(inputv);
	}

}

/******************************************************************************
 * Host main routine
 */
void activate2cuda(tensor_t<float> in, tensor_t<float> weights,
		std::vector<float> input, tensor_t<float> out) {

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

	blocksPerGrid = std::min(deviceProp.multiProcessorCount, 2);
	int cudaCores = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
	threadsPerBlock = deviceProp.maxThreadsPerBlock;
	totalThreads = blocksPerGrid * threadsPerBlock;

	cudaError_t err = cudaSuccess;

	h_in = &in;
	int in_mem_size = sizeof(in);

	// TODO remove
	printf("sizeof(in) , sizeof(out) = %lu * %lu \n", sizeof(in), sizeof(in));

	if (h_in == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **) &d_in, in_mem_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_in, h_in, in_mem_size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr,
				"Failed to copy vector C from device to host (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy weights

	h_weights = &weights;
	int weights_mem_size = sizeof(weights);

	if (h_weights == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **) &d_weights, weights_mem_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_weights, h_weights, weights_mem_size,
			cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr,
				"Failed to copy vector C from device to host (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Lanzar KERNEL

	printf("CUDA kernel launch with %d blocks of %d threads. Total: %i\n",
			blocksPerGrid, threadsPerBlock, totalThreads);

	printf("Tensor in: \n");
	print_tensor(*h_in);

	activate_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_weights, d_input,
			d_out);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// get input array

	h_input = &input[0];
	int input_mem_size = sizeof(float) * input.size();

	err = cudaMemcpy(h_input, d_input, input_mem_size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr,
				"Failed to copy vector INPUT from device to host (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_input);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to free device vector INPUT (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Get out
	h_out = &out;
	int out_mem_size = sizeof(out);

	err = cudaMemcpy(h_out, d_out, out_mem_size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr,
				"Failed to copy OUT from device to host (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_out);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to free device OUT (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free host memory

	//free(h_cases);
	err = cudaDeviceReset();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to deinitialize the device! error=%s\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// TODO remove
	printf("cuda out: ");
}

